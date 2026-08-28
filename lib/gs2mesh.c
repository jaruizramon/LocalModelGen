/* gs2mesh.c - procedural Gaussian-splat -> mesh, pure CPU (no AI/CUDA/xformers).
 * Reads a TRELLIS save_ply gaussian (17 floats/splat: x,y,z / f_dc_0..2 /
 * opacity / scale_0..2 / rot_0..3), scatters each splat's 3D Gaussian density
 * into an (N+1)^3 node grid (max-density), extracts the isosurface via marching
 * tetrahedra, writes an OBJ.  Runs in CPU RAM only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef GRID_N
#define GRID_N 256
#endif
#define CUTOFF 2.5f

typedef struct { float pos[3], col[3], op, sc[3], q[4]; } Splat;
typedef struct { float v[3]; unsigned char r, g, b; } Vtx;

static Vtx  *verts;  static size_t nverts, capv;
static size_t *tris; static size_t ntris, capt;
static int N;
static float *nodes;
static float mn[3], mx[3];

static void pushv(float x,float y,float z,float r,float g,float b){
    if(nverts==capv){capv=capv?capv*2:1<<20; verts=realloc(verts,capv*sizeof(Vtx));}
    verts[nverts].v[0]=x;verts[nverts].v[1]=y;verts[nverts].v[2]=z;
    verts[nverts].r=(unsigned char)(r*255);verts[nverts].g=(unsigned char)(g*255);verts[nverts].b=(unsigned char)(b*255);
    nverts++;
}
static void pushf(size_t a,size_t b,size_t c){
    if(ntris==capt){capt=capt?capt*2:1<<20; tris=realloc(tris,capt*3*sizeof(size_t));}
    tris[ntris*3]=a;tris[ntris*3+1]=b;tris[ntris*3+2]=c; ntris++;
}
static void quat_to_R(const float q[4], float R[9]){
    float w=q[0],x=q[1],y=q[2],z=q[3];
    R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z);   R[2]=2*(x*z+w*y);
    R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
    R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}
static inline float node_val(int a,int b,int c){
    return nodes[((size_t)a*(N+1)+b)*(N+1)+c];
}

int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: %s in.ply out.obj [gridN] [threshold_frac]\n",argv[0]); return 2; }
    const char *ply=argv[1], *out=argv[2];
    N = argc>3?atoi(argv[3]):GRID_N;
    float THFrac = argc>4?(float)atof(argv[4]):0.35f;

    FILE *f=fopen(ply,"rb"); if(!f){perror(ply);return 1;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    unsigned char *buf=malloc(sz); fread(buf,1,sz,f); fclose(f);
    long idx=-1; for(long i=0;i+9<sz;i++) if(memcmp(buf+i,"end_header",10)==0){idx=i+11;break;}
    if(idx<0){fprintf(stderr,"no end_header\n");return 1;}
    long n=0; { char *p=strstr((char*)buf,"element vertex"); if(p) sscanf(p,"element vertex %ld",&n); }
    fprintf(stderr,"DEBUG n=%ld idx=%ld sz=%ld\n",n,idx,sz);

    Splat *sp=malloc(n*sizeof(Splat));
    const float shC0=0.28209479177387814f;
    for(long i=0;i<n;i++){ const float *v=(float*)(buf+idx+i*17*4);
        sp[i].pos[0]=v[0];sp[i].pos[1]=v[1];sp[i].pos[2]=v[2];
        sp[i].op=1.0f/(1.0f+expf(-v[9]));
        sp[i].sc[0]=expf(v[10]);sp[i].sc[1]=expf(v[11]);sp[i].sc[2]=expf(v[12]);
        sp[i].q[0]=v[13];sp[i].q[1]=v[14];sp[i].q[2]=v[15];sp[i].q[3]=v[16];
        for(int k=0;k<3;k++){ float cc=0.5f+shC0*v[6+k]; sp[i].col[k]=cc<0?0:(cc>1?1:cc); }
    }
    for(int k=0;k<3;k++){mn[k]=1e9f;mx[k]=-1e9f;}
    for(long i=0;i<n;i++) for(int k=0;k<3;k++){
        if(sp[i].pos[k]<mn[k])mn[k]=sp[i].pos[k]; if(sp[i].pos[k]>mx[k])mx[k]=sp[i].pos[k];
    }
    for(int k=0;k<3;k++) if(mx[k]-mn[k]<1e-6f) mx[k]=mn[k]+1e-3f;

    size_t nnode=(size_t)(N+1)*(N+1)*(N+1);
    nodes=calloc(nnode,sizeof(float));
    float mxden=0.f;
    // omp off (race on nodes)
    for(long i=0;i<n;i++){
        float R[9]; quat_to_R(sp[i].q,R);
        float Sinv[9];
        for(int a=0;a<3;a++) for(int b=0;b<3;b++){
            float s=0; for(int m=0;m<3;m++) s+=R[a*3+m]*R[b*3+m]/(sp[i].sc[m]*sp[i].sc[m]);
            Sinv[a*3+b]=s;
        }
        float scmax=sp[i].sc[0];
        if(sp[i].sc[1]>scmax)scmax=sp[i].sc[1]; if(sp[i].sc[2]>scmax)scmax=sp[i].sc[2];
        float extent=CUTOFF*scmax;
        int lo[3],hi[3];
        for(int k=0;k<3;k++){
            float spn=N*(sp[i].pos[k]-mn[k])/(mx[k]-mn[k]);
            float pad=extent*N/(mx[k]-mn[k]);
            lo[k]=(int)floorf(spn-pad); if(lo[k]<0)lo[k]=0;
            hi[k]=(int)ceilf(spn+pad);  if(hi[k]>N)hi[k]=N;
        }
        for(int a=lo[0];a<=hi[0];a++) for(int b=lo[1];b<=hi[1];b++) for(int c=lo[2];c<=hi[2];c++){
            float d[3];
            d[0]=(mn[0]+((float)a/N)*(mx[0]-mn[0]))-sp[i].pos[0];
            d[1]=(mn[1]+((float)b/N)*(mx[1]-mn[1]))-sp[i].pos[1];
            d[2]=(mn[2]+((float)c/N)*(mx[2]-mn[2]))-sp[i].pos[2];
            float q=d[0]*(Sinv[0]*d[0]+Sinv[1]*d[1]+Sinv[2]*d[2])
                   +d[1]*(Sinv[3]*d[0]+Sinv[4]*d[1]+Sinv[5]*d[2])
                   +d[2]*(Sinv[6]*d[0]+Sinv[7]*d[1]+Sinv[8]*d[2]);
            float dens=sp[i].op*expf(-0.5f*q);
            size_t cell=((size_t)a*(N+1)+b)*(N+1)+c;
            if(dens>nodes[cell]) nodes[cell]=dens;
            if(dens>mxden) mxden=dens;
        }
    }
    float TH=mxden*THFrac;
    printf("grid=%d splats=%ld max_density=%.4f thr=%.5f\n",N,n,mxden,TH);

    /* marching tetrahedra */
    static const int tets[6][4]={{0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6},{0,5,1,6}};
    #pragma omp parallel for schedule(dynamic)
    for(int a=0;a<N;a++) for(int b=0;b<N;b++) for(int c=0;c<N;c++){
        float v[8]={ node_val(a,b,c),node_val(a+1,b,c),node_val(a+1,b+1,c),node_val(a,b+1,c),
                     node_val(a,b,c+1),node_val(a+1,b,c+1),node_val(a+1,b+1,c+1),node_val(a,b+1,c+1) };
        float P[8][3];
        for(int m=0;m<8;m++){ int ax=(m&1)?1:0, by=(m&2)?1:0, cz=(m&4)?1:0;
            P[m][0]=mn[0]+((a+ax)/(float)N)*(mx[0]-mn[0]);
            P[m][1]=mn[1]+((b+by)/(float)N)*(mx[1]-mn[1]);
            P[m][2]=mn[2]+((c+cz)/(float)N)*(mx[2]-mn[2]);
        }
        for(int t=0;t<6;t++){
            const int *edx=tets[t];
            float dv[4]={v[edx[0]],v[edx[1]],v[edx[2]],v[edx[3]]};
            unsigned mask=0; for(int m=0;m<4;m++) if(dv[m]>TH) mask|=1u<<m;
            if(mask==0||mask==15) continue;
            static const int pe[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
            size_t thev[6]; int en=0;
            for(int e=0;e<6;e++){
                int na=edx[pe[e][0]], nb=edx[pe[e][1]];
                if((v[na]>TH)!=(v[nb]>TH)){
                    float fa=(v[na]-TH)/(v[na]-v[nb]);
                    float x=P[na][0]+fa*(P[nb][0]-P[na][0]);
                    float y=P[na][1]+fa*(P[nb][1]-P[na][1]);
                    float z=P[na][2]+fa*(P[nb][2]-P[na][2]);
                    pushv(x,y,z,0.55f,0.5f,0.45f);
                    thev[en++]=nverts-1;
                }
            }
            if(en==3) pushf(thev[0],thev[1],thev[2]);
            else if(en==4){ pushf(thev[0],thev[1],thev[2]); pushf(thev[0],thev[2],thev[3]); }
        }
    }
    free(sp);free(buf);

    /* ---- weld + decimate: vertex clustering (merges coincident MT verts into
     * a connected surface + drops the density) ---- */
    if(nverts>0){
        float cell=(mx[0]-mn[0])/N*2.0f; /* cluster size ~ 2 MT cells */
        if(cell<1e-6f) cell=1e-3f;
        typedef struct { long k[3]; size_t oi, ci; } QV;
        QV *qs=malloc(nverts*sizeof(QV));
        for(size_t i=0;i<nverts;i++){ qs[i].k[0]=(long)lroundf(verts[i].v[0]/cell);
            qs[i].k[1]=(long)lroundf(verts[i].v[1]/cell); qs[i].k[2]=(long)lroundf(verts[i].v[2]/cell);
            qs[i].oi=i; qs[i].ci=0; }
        /* sort by k */
        for(size_t i=1;i<nverts;i++){ QV q=qs[i]; size_t j=i;
            while(j>0 && (qs[j-1].k[0]>q.k[0] || (qs[j-1].k[0]==q.k[0] && (qs[j-1].k[1]>q.k[1] || (qs[j-1].k[1]==q.k[1] && qs[j-1].k[2]>q.k[2]))))){ qs[j]=qs[j-1]; j--; } qs[j]=q; }
        size_t clusters=0; for(size_t i=0;i<nverts;i++){ if(i==0 || qs[i].k[0]!=qs[i-1].k[0] || qs[i].k[1]!=qs[i-1].k[1] || qs[i].k[2]!=qs[i-1].k[2]) clusters++; qs[i].ci=clusters-1; }
        Vtx *nv2=malloc(clusters*sizeof(Vtx));
        size_t *remap=calloc(nverts,sizeof(size_t));
        for(size_t i=0;i<nverts;i++){ remap[qs[i].oi]=qs[i].ci; nv2[qs[i].ci]=verts[qs[i].oi]; }
        free(qs);
        size_t *t2=malloc(ntris*3*sizeof(size_t)); size_t nt2=0;
        for(size_t i=0;i<ntris;i++){ size_t a=remap[tris[i*3]],b=remap[tris[i*3+1]],c=remap[tris[i*3+2]];
            if(a!=b&&b!=c&&a!=c){ t2[nt2*3]=a;t2[nt2*3+1]=b;t2[nt2*3+2]=c; nt2++; } }
        free(remap); free(tris); tris=t2; ntris=nt2;
        free(verts); verts=nv2; nverts=clusters;
        printf("welded/decimated: %zu verts, %zu tris\n",nverts,ntris);
    }

    /* ---- Taubin smoothing (blend the splat bumps, no shrinkage) ---- */
    if(nverts>0 && ntris>0){
        /* build per-vertex adjacency (one ring) */
        size_t *deg=calloc(nverts,sizeof(size_t));
        for(size_t i=0;i<ntris;i++){ deg[tris[i*3]]++; deg[tris[i*3+1]]++; deg[tris[i*3+2]]++; }
        size_t *off=calloc(nverts+1,sizeof(size_t));
        for(size_t i=0;i<nverts;i++) off[i+1]=off[i]+deg[i];
        size_t *nbr=malloc(off[nverts]*sizeof(size_t)); size_t *fill=malloc(nverts*sizeof(size_t));
        memcpy(fill,deg,nverts*sizeof(size_t)); /* reuse as write cursors */
        { /* rebuild cursors from offsets */
            for(size_t i=1;i<=nverts;i++) off[i]=off[i-1]+ (i>0?deg[i-1]:0); }
        for(size_t v=0;v<nverts;v++) fill[v]=off[v];
        for(size_t i=0;i<ntris;i++){ size_t a=tris[i*3],b=tris[i*3+1],c=tris[i*3+2];
            nbr[fill[a]++]=b; nbr[fill[a]++]=c; nbr[fill[b]++]=a; nbr[fill[b]++]=c; nbr[fill[c]++]=a; nbr[fill[c]++]=b; }
        free(fill);
        float *pos=malloc(nverts*3*sizeof(float));
        for(size_t i=0;i<nverts;i++){ pos[i*3]=verts[i].v[0];pos[i*3+1]=verts[i].v[1];pos[i*3+2]=verts[i].v[2]; }
        float *ppos=malloc(nverts*3*sizeof(float));
        float lambda=0.5f, mu=-0.53f;
        for(int iter=0;iter<16;iter++){
            for(size_t v=0;v<nverts;v++){ float ax=0,ay=0,az=0; size_t d=off[v+1]-off[v];
                if(!d) continue; for(size_t k=off[v];k<off[v+1];k++){ ax+=pos[nbr[k]*3]; ay+=pos[nbr[k]*3+1]; az+=pos[nbr[k]*3+2]; }
                ax/=d; ay/=d; az/=d; float s=(iter%2==0)?lambda:mu;
                ppos[v*3]=pos[v*3]+s*(ax-pos[v*3]); ppos[v*3+1]=pos[v*3+1]+s*(ay-pos[v*3+1]); ppos[v*3+2]=pos[v*3+2]+s*(az-pos[v*3+2]); }
            float *t=pos; pos=ppos; ppos=t;
        }
        for(size_t i=0;i<nverts;i++){ verts[i].v[0]=pos[i*3];verts[i].v[1]=pos[i*3+1];verts[i].v[2]=pos[i*3+2]; }
        free(pos);free(ppos);free(nbr);free(off);free(deg);
        printf("taubin smoothed: %zu verts\n",nverts);
    }

    FILE *o=fopen(out,"w"); if(!o){perror(out);return 1;}
    fprintf(o,"# gs2mesh procedural\n");
    /* match the splat view frame: ply is (x,y,z); splat uses (x,-y,-z) */
    for(size_t i=0;i<nverts;i++) fprintf(o,"v %.6f %.6f %.6f\n",verts[i].v[0],-verts[i].v[1],-verts[i].v[2]);
    for(size_t i=0;i<ntris;i++) fprintf(o,"f %zu %zu %zu\n",tris[i*3]+1,tris[i*3+1]+1,tris[i*3+2]+1);
    printf("wrote %s: %zu verts, %zu tris\n",out,nverts,ntris);
    fclose(o);
    return 0;
    fprintf(o,"# gs2mesh procedural\n");
    for(size_t i=0;i<nverts;i++) fprintf(o,"v %.6f %.6f %.6f\n",verts[i].v[0],verts[i].v[1],verts[i].v[2]);
    for(size_t i=0;i<ntris;i++) fprintf(o,"f %zu %zu %zu\n",tris[i*3]+1,tris[i*3+1]+1,tris[i*3+2]+1);
    printf("wrote %s: %zu verts, %zu tris\n",out,nverts,ntris);
    fclose(o);
    return 0;
}
