# Han-Nom-Recognition-System
Convert **Keras** models to **TFJS** and run it in the browser.

**Note: using latest version of the browser, we're using Google Chrome >= 112**
[Click here]([http://210.211.125.36/])

## Requirements

### For Python

See in requirements.txt

### For Javascript

Bootstrap, Dropzone, Pyodide, TensorflowJS

## Setup / Explain

In this project, we have 2 models: **Detection** and **Recognition**

### 1. Convert Keras models to TFJS

First, download the weights and put it into **weights** folder by [Click here](https://github.com/ds4v/NomNaOCR?tab=readme-ov-file#1-quy-tr%C3%ACnh-hu%E1%BA%A5n-luy%E1%BB%87n) and run the command below to convert to TFJS models.

```shell
python convert_to_tfjs.py
```

The result will be saved at **web/assets/weights** folder.


### 2. Download Pyodide

Download & put **pyodide** folder to **web/assets** folder.

[https://github.com/pyodide/pyodide/releases/download/0.26.0a4/pyodide-0.26.0a4.tar.bz2](https://github.com/pyodide/pyodide/releases/download/0.26.0a4/pyodide-0.26.0a4.tar.bz2)

Because almost image processing functions can't work or can't convert to **Javascript** language. We have to install **Pyodide** to use alongside.

### 3. Other libraries

We're using **Bootstrap** as a based theme of the web and integrating **Dropzone** javascript library for the Drag & Drop image area.

### 4. Process

### 4.1 Intialization

First, we load **Pyodide** and models **Detection**, **Recognition**. 

Second, install more **Python** packages like: **OpenCV**, **pyclipper**, **shapely**, **scikit-image** using **micropip**.

Final, load custom Python functions in the folder **web/assets/scripts**.

```js
let pyodide = await loadPyodide();

const DBNetModel = await tf.loadGraphModel('assets/weights/DBNet/model.json');

const CRNNModel = await tf.loadGraphModel('assets/weights/CRNN/model.json');

await pyodide.loadPackage('micropip')
let micropip = pyodide.pyimport('micropip')

await micropip.install('opencv-python');
await micropip.install('pyclipper');
await micropip.install('shapely');
await micropip.install('scikit-image');

// Load custom functions
await loadProsessor(pyodide);
await loadFunctions(pyodide);
```

And then, we setup **Dropzone** and attach **click** event to the **START PROCESS** button.

```js
let dropzone = new Dropzone("div#fileArea", {
    url: '/',
    disablePreviews: false,
    uploadMultiple: false,
    autoProcessQueue: false,
    addRemoveLinks: true,
    maxFiles: 1,
    acceptedFiles: "image/*"
});

startButton.addEventListener('click', function() {
    ...
});
```

### 4.2 Detection

When a guest choose or drop an image to the **Dropzone** area and click **START PROCESS**. We'll convert the image to a **Tensor** and use **Pyodide** to preprocessing the image.

```js
let raw_tensor = tf.browser.fromPixels(img, 3);
let tensor = await raw_tensor.array();

tensor = JSON.stringify(tensor);
tensor = JSON.parse(pyodide.runPython(`
import json
db_input = json.loads('${tensor}')
db_input = np.asarray(db_input, dtype=np.uint8)
db_input = resize_image_short_side(db_input).astype(float) / 255.0
json.dumps(db_input.tolist())
`))
```

And then, we'll convert the processed image to a **Tensor** again.

```js
const dpnTensor = tf.tensor(tensor, undefined, "float32")
// (height, width, 3)
```

Final, we'll expand first dimension of the **Tensor** to get the shape **(1, height, width, 3)** using **tf.expandDims()**.
And put it as an input for the model.

```js
const detResult = DBNetModel.predict({image: dpnTensor.expandDims()});
```

**Output** of the model is an array includes: **probability_map**, **threshold_map**, **approximate_binary_map**. We'll use only **probability_map**.

### 4.3 Recognition

We will use **PostProcessor** class to convert **probability_map** to an array of boxes. Each box is a quads (x1, y1, x2, y2, x3, y3, x4, y4).

```js
// detResult[0] is probability_map
const result = await detResult[0].array();
const resultStr = JSON.stringify(result);
const boxes = JSON.parse(pyodide.runPython(`
processor = PostProcessor(min_box_score=0.5, max_candidates=1000)
arr = json.loads('${resultStr}')
batch_boxes, batch_scores = processor(np.asarray(arr, dtype=np.float32), [(${img.height}, ${img.width})])
boxes = order_boxes4nom(batch_boxes[0])
json.dumps([box.tolist() for box in boxes])
`));
```

Now, we'll crop an area of the image following box positions and define a new variable **patches**

```js
let patches = JSON.parse(pyodide.runPython(`
patches = []
arr = json.loads('${JSON.stringify(raw_image)}')
raw_image = np.asarray(arr, dtype=np.uint8)
for idx, box in enumerate(boxes):
    patch = get_patch(raw_image, box)                       
    patch = distortion_free_resize(patch, align_top=True)
    patch = patch.astype(np.float32) / 255.0
    patches.append(patch)
json.dumps([patch.tolist() for patch in patches])
`)); 
```

**Patches** now is an array includes images. We'll use a loop and process each patch, convert the patch to a shape 
**(1, height, width, 3)** and put it as an input for the model.

```js
for(const patch of patches) {
    ...

    const patchTensor = tf.tensor(patch, undefined, "float32");
    const recResult = await CRNNModel.executeAsync({image: patchTensor.expandDims()});
    const pred_tokens = await recResult.array();
}
```

We'll use **CTCGreedyDecoder** to decode the ouput and find words by the index.

```js
const decoded = await ctcGreedyDecoder(pred_tokens[0]);
let texts = await tokens2texts(decoded, num2char);
texts = texts.filter(x => x !== 'undefined').join('');    
console.log(texts) // 𡦂才𡦂命窖󰑼恄饒
```

All functions you can find in the folder: **web/assets/js/**.

## Reference

We've using pretrained **weights** from **Original** repo and image processing code from **Demo** repo.

Original: [https://github.com/ds4v/NomNaOCR](https://github.com/ds4v/NomNaOCR)

Demo: [https://github.com/ds4v/NomNaSite](https://github.com/ds4v/NomNaSite)
