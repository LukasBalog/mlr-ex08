# Leaf classifier — live camera demo (MPA-MLR, exercise 08)

**Live page: <https://lukasbalog.github.io/mlr-ex08/>**

Point your phone's rear camera at a leaf and the four class probabilities update
live. Everything runs in the browser — the image never leaves the device.

Open the page on a phone either by typing the URL or by scanning the QR code at
the bottom of the page (generated in the browser by
[qrcode-generator](https://www.npmjs.com/package/qrcode-generator), not by a QR
web service).

## What it does

`index.html` (plain JavaScript, no build step) loads `leaf_model.onnx` with
**onnxruntime-web 1.23.2** from a CDN and runs it on the WebAssembly CPU backend
(`executionProviders: ["wasm"]`), so the result does not depend on a GPU or on
which vendor's tensor cores the browser happens to find.

Preprocessing follows `model_info.json` exactly:

| step | detail |
| --- | --- |
| crop | the **central square** of the frame, side = `min(width, height)` |
| resize | to `input_size`² = 96 × 96 |
| scale | RGB bytes → 0–1 |
| standardise | per channel with `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` |
| tensor | `1 × 3 × 96 × 96` float32, NCHW, input name `image` |

The model's output tensor is named `logits`, so softmax is applied in the page to
turn them into the displayed probabilities.

The preview is a square box with `object-fit: cover`, which displays exactly the
central square that is fed to the network — what you see is what the model sees.

A first load transfers **8.29 MB** in total: 5.67 MB for `leaf_model.onnx`,
2.58 MB for `ort-wasm-simd-threaded.wasm` (11.35 MB on disk, compressed in
transit) and 24 KB for the JavaScript glue. Counted off the browser's network
log. Worth knowing on mobile data; the cache spares you afterwards.

Classes: `apple`, `cherry`, `chestnut`, `maple`.

## Verification

`check_image.jpg` must reproduce `check_expected.json`. The page runs this itself
on load and prints the table in the **Status** card ("Re-run" repeats it).
Checked on 2026-11-13 against the published URL in headless Chrome driven over
the DevTools protocol — the page reported **PASS**, max |difference| **4.71e-7**:

| class | this page | expected | \|diff\| |
| --- | --- | --- | --- |
| apple | 0.034427 | 0.034427 | 4.4e-7 |
| cherry | 0.965568 | 0.965567 | 4.7e-7 |
| chestnut | 0.000001 | 0.000001 | 8.3e-12 |
| maple | 0.000004 | 0.000004 | 6.7e-11 |

It is **not** a decoding difference: Chrome's decoded pixels for `check_image.jpg`
were compared against PIL's byte for byte and all 27 648 of them are equal (the
image is already 96×96, so nothing gets resampled), and the two float32 input
tensors agree to the last digit printed. What differs is the arithmetic inside
the two runtimes — the same model fed the same tensor gives logits that disagree
by at most **9.3e-6** between the WebAssembly build and the native CPU build,
which is ordinary float32 accumulation order, and softmax shrinks that to the
~5e-7 seen above.

The Python reference run (`onnxruntime` 1.23.2 + PIL, same pipeline) lands
**9.9e-7** from `check_expected.json` — i.e. the browser is no further from the
expected numbers than a second native run is. It reports the input as `image` and
the output as `logits`, as the page assumes.

In the same run the two interactive paths were exercised against the live URL:
with a synthetic camera device the loop reached 7.5 fps and printed a verdict
every frame, "Stop" released the stream, and feeding `check_image.jpg` through
**Classify a photo…** returned `cherry` at 96.6%. The QR card drew a real inline
`<svg>`, so no third-party image service is involved.

## Files

| file | role |
| --- | --- |
| `index.html` | the whole app — camera loop, photo upload, results, self-test, QR |
| `leaf_model.onnx` | the classifier, 1×3×96×96 → 4 logits |
| `model_info.json` | input size, mean, std, class names — read at run time |
| `check_image.jpg` | 96×96 verification image |
| `check_expected.json` | its expected probabilities |

## Using it

* **Start rear camera** — requests `facingMode: environment`, falls back to any
  camera, and holds at ≈8 inferences/s because the loop takes at most one frame
  per 120 ms. Measured in headless Chrome on a desktop CPU: 7.5 fps at 4.9–7.4 ms
  per inference. The page always prints the millisecond figure it measured, not
  this one.
* **Classify a photo…** — fallback that needs no camera permission. The chosen
  file goes through the identical crop/resize/normalise path.
* **Status** — model download progress, camera state and the self-test result.

### Safari on iPhone

* HTTPS is required for `getUserMedia`; GitHub Pages provides it.
* The camera only starts from a tap, which is why it is behind a button.
* `<video playsinline muted>` keeps the preview in the page instead of
  fullscreen on iOS.
* WASM threads are off (`numThreads = 1`), because GitHub Pages sends no
  COOP/COEP headers and therefore no `SharedArrayBuffer` — checked against the
  live response, which carries only `Access-Control-Allow-Origin` and
  `Cache-Control`.
* Needs a browser with **WebAssembly SIMD**: onnxruntime-web 1.23.2 ships only
  `ort-wasm-simd-*.wasm`, so there is no non-SIMD build to fall back on. ONNX
  Runtime's own compatibility table marks the WASM CPU backend as working in
  Safari on iOS but names no minimum version, so none is claimed here. Where the
  runtime cannot start, the page still loads, photo mode still works, and the
  Status card reports the error.

These points follow from documented Safari behaviour and from the headers GitHub
Pages actually sends; they were not executed on an iPhone. Only the Chrome path
was run — see Verification.

## Repo

Served by GitHub Pages from the `main` branch, at the repository root — Pages
needs the app there. The course notebook is deliberately left out (see
`.gitignore`): only the files the page loads are committed.
