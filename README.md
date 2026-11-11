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
**onnxruntime-web 1.23.2** from a CDN and runs it on the WebAssembly CPU backend,
which gives the same numbers in every browser.

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

Classes: `apple`, `cherry`, `chestnut`, `maple`.

## Verification

`check_image.jpg` must reproduce `check_expected.json`. The page runs this itself
on load and prints the table in the **Status** card ("Re-run" repeats it). In
Chrome it reports **PASS**, max |difference| ≈ 5e-7:

| class | this page | expected |
| --- | --- | --- |
| apple | 0.034427 | 0.034427 |
| cherry | 0.965568 | 0.965567 |
| chestnut | 0.000001 | 0.000001 |
| maple | 0.000004 | 0.000004 |

A reference run in Python (`onnxruntime` 1.23.2 + PIL, same pipeline as above)
agrees with `check_expected.json` to 1.0e-6.

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
  camera, and runs ~8 inferences/s (≈8 ms each on a phone).
* **Classify a photo…** — fallback that needs no camera permission. The chosen
  file goes through the identical crop/resize/normalise path.
* **Status** — model download progress, camera state and the self-test result.

### Safari on iPhone

* HTTPS is required for `getUserMedia`; GitHub Pages provides it.
* The camera only starts from a tap, which is why it is behind a button.
* `<video playsinline muted>` keeps the preview in the page instead of
  fullscreen on iOS.
* WASM threads are off (`numThreads = 1`), because GitHub Pages sends no
  COOP/COEP headers and therefore no `SharedArrayBuffer`.
* Needs **iOS 16.4+** for WebAssembly SIMD; on older versions the page still
  loads and photo mode works, and the Status card explains what failed.

## Repo

Served by GitHub Pages from the `main` branch.
