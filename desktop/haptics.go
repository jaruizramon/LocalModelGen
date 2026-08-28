package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os/exec"
)

// beep emits an audible cue per stage of a generation. Audio is best-effort: a
// generated 16-bit mono WAV is piped to paplay/aplay; if neither exists or there
// is no sound server, the terminal BEL character still fires. Call from the UI
// goroutine; playback is async so it never blocks the render loop.
func beep(kind string) {
	fmt.Print("\a") // terminal bell fallback
	freq, dur := 660.0, 120
	switch kind {
	case "start":
		freq, dur = 520, 90
	case "done":
		freq, dur = 990, 170
	case "error":
		freq, dur = 330, 360
	}
	go playTone(freq, dur)
}

// playTone pipes a freshly built WAV to the first available audio player. Errors
// are ignored: haptics should never take down the app.
func playTone(freq float64, durMs int) {
	wav := wavTone(freq, durMs)
	for _, p := range []string{"paplay", "aplay", "pw-play"} {
		if _, err := exec.LookPath(p); err != nil {
			continue
		}
		cmd := exec.Command(p, "-")
		cmd.Stdin = bytes.NewReader(wav)
		if err := cmd.Run(); err == nil {
			return
		}
	}
}

// wavTone builds a 16-bit mono PCM WAV of a sine at freq for durMs, with a short
// attack/release envelope so the tone does not click on/off.
func wavTone(freq float64, durMs int) []byte {
	const sr = 44100
	n := sr * durMs / 1000
	var buf bytes.Buffer
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, uint32(36+n*2))
	buf.WriteString("WAVEfmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))       // fmt chunk size
	binary.Write(&buf, binary.LittleEndian, uint16(1))        // PCM
	binary.Write(&buf, binary.LittleEndian, uint16(1))        // mono
	binary.Write(&buf, binary.LittleEndian, uint32(sr))       // sample rate
	binary.Write(&buf, binary.LittleEndian, uint32(sr*2))     // byte rate
	binary.Write(&buf, binary.LittleEndian, uint16(2))        // block align
	binary.Write(&buf, binary.LittleEndian, uint16(16))       // bits per sample
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(n*2))
	const atk, rel = 0.005 * sr, 0.008 * sr // 5ms attack, 8ms release
	for i := 0; i < n; i++ {
		env := 1.0
		if float64(i) < atk {
			env = float64(i) / atk
		}
		if float64(n-i) < rel {
			env = float64(n-i) / rel
		}
		s := math.Sin(2*math.Pi*freq*float64(i)/sr) * 0.5 * env
		binary.Write(&buf, binary.LittleEndian, int16(s*32767))
	}
	return buf.Bytes()
}
