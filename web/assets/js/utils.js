// Load code from processor.py, functions.py and run with pyodide
async function loadProsessor(pyodide) {
    let response = await fetch("assets/scripts/processor.py");
    let body = await response.text();
    await pyodide.runPythonAsync(body);
}

async function loadFunctions(pyodide) {
    let response = await fetch("assets/scripts/functions.py");
    let body = await response.text();
    await pyodide.runPythonAsync(body);
}

/**
 * Resize an image tensor by its shorter side to a specified size.
 * @param {tf.Tensor3D} imageTensor - The image tensor to resize.
 * @param {number} imageShortSide - The target size for the image's short side.
 * @returns {tf.Tensor3D} The resized image tensor.
 */

// Resize an image tensor before put it into a model -> increase model's computational performance
async function resizeImageShortSide(imageTensor, imageShortSide = 736) {
    // Ensure the tensor is 3D (height, width, channels)
    if (imageTensor.shape.length !== 3) {
      throw new Error('Image tensor must be 3D (height, width, channels).');
    }
  
    const [height, width, _] = imageTensor.shape;
    let newHeight, newWidth;
  
    // Determine new width and height maintaining the aspect ratio
    if (height < width) {
      newHeight = imageShortSide;
      newWidth = parseInt(Math.round(newHeight / height * width / 32) * 32);
    } else {
      newWidth = imageShortSide;
      newHeight = parseInt(Math.round(newWidth / width * height / 32) * 32);
    }
  
    // Resize the image
    const resizedImage = tf.image.resizeBilinear(imageTensor, [newHeight, newWidth], true);
  
    return resizedImage;
}

function ctcGreedyDecoder(probs, blankLabel = 0) {
    // Convert the 2D array of probabilities to a tensor
    const probsTensor = tf.tensor(probs);
    
    // Use argMax to find the index of the maximum value in the tensor along axis 1
    const argMaxIndices = probsTensor.argMax(1);
    
    // Convert tensor to array to process the decoding
    return argMaxIndices.array().then(indices => {
      const decodedSequence = [];
      let prevLabel = null;
  
      // Iterate over the indices and construct the decoded sequence
      indices.forEach((label, index) => {
        if (label !== blankLabel && label !== prevLabel) {
          decodedSequence.push(label);
        }
        prevLabel = label;
      });
  
      return decodedSequence;
    });
}

// Converts an array of tokens into an array of corresponding text
async function tokens2texts(batchTokens, num2char) {
    let batchTexts = [];

    for (let token of batchTokens) {
        let text = '';
        
        if (token !== 0 && token !== -1) {
          text += num2char[token];
        }

        batchTexts.push(text);
    }
  
    return batchTexts;
}