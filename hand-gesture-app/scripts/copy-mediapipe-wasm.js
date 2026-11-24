// Copies MediaPipe's WebAssembly runtime from node_modules into public/,
// so the app loads it from our own server instead of a CDN.
// Runs automatically after `npm install` (see "postinstall" in package.json).
import { cpSync } from 'node:fs'

cpSync('node_modules/@mediapipe/tasks-vision/wasm', 'public/mediapipe/wasm', { recursive: true })
console.log('Copied MediaPipe wasm files to public/mediapipe/wasm')
