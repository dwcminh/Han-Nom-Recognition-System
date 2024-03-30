(async function() {
    Dropzone.autoDiscover = false;

    // Build Python environment on the browser
    let pyodide = await loadPyodide();

    // Load models
    const DBNetModel = await tf.loadGraphModel('assets/weights/DBNet/model.json');
    const CRNNModel = await tf.loadGraphModel('assets/weights/CRNN/model.json');

    // Define HTML elements
    const startButton = document.getElementById('startProcessBtn');
    const predictBody = document.getElementById('predictBody');
    const loading = document.getElementById('loading');
    
    // Install Python external packages (supporting for image processing)
    await pyodide.loadPackage('micropip')
    let micropip = pyodide.pyimport('micropip')

    await micropip.install('opencv-python');
    await micropip.install('pyclipper');
    await micropip.install('shapely');
    await micropip.install('scikit-image');

    // Load custom functions
    await loadProsessor(pyodide);
    await loadFunctions(pyodide);


    // Define dropzone, the user can drop or click to the area to Upload File
    let dropzone = new Dropzone("div#fileArea", {
        url: '/',
        disablePreviews: false,
        uploadMultiple: false,
        autoProcessQueue: false,
        addRemoveLinks: true,
        maxFiles: 1,
        acceptedFiles: "image/*"
    });
    
    // Add click event to the START PROCESS button
    startButton.addEventListener('click', function() {
        startButton.classList.add('disabled');
        startButton.setAttribute('disabled', 'disabled');
        loading.classList.add('active');

        // Check the file exists or not
        if(dropzone.getQueuedFiles().length <= 0) {
            alert("Please select at least 1 image");
            startButton.classList.remove('disabled');
            startButton.removeAttribute('disabled');         
            loading.classList.remove('active'); 
            return;
        }

        // Get the file, convert it to the URL and create element
        const files = dropzone.getQueuedFiles();
        const file = files[0];
        const img = document.createElement('img')
        img.src = URL.createObjectURL(file);

        img.onload = async () => {
            // Convert the image to a tensor and then convert the tensor to array 
            // Ref: https://www.geeksforgeeks.org/tensorflow-js-tf-browser-frompixels-function/
            let raw_tensor = tf.browser.fromPixels(img, 3);
            let tensor = await raw_tensor.array();

            // Convert array to JSON string and process the JSON string in Python environment
            tensor = JSON.stringify(tensor);
            tensor = JSON.parse(pyodide.runPython(`
            import json
            db_input = json.loads('${tensor}')
            db_input = np.asarray(db_input, dtype=np.uint8)
            db_input = resize_image_short_side(db_input).astype(float) / 255.0
            json.dumps(db_input.tolist())
            `))
            
            // Convert processed image to a Tensor and use this Tensor as a input for Detection model
            // Now the tensor is a shape (height, width, 3) we need to use tf.expandDims() 
            // to convert it to the shape (1, height, width, 3) it's mean (batch size, height, width, 3)
            const dpnTensor = tf.tensor(tensor, undefined, "float32")
            const detResult = DBNetModel.predict({image: dpnTensor.expandDims()});

            if(detResult.length > 0) {
                // Convert Detection result to an array to JSON string
                // keep using JSON string for the image processing 
                // and we need to use PostProcessor to the get the (X1, Y1, X2, Y2, X3, Y3, X4, Y4) positions on the image
                // ref: https://github.com/ds4v/NomNaSite/blob/main/processor.py
                const result = await detResult[0].array();
                const resultStr = JSON.stringify(result);
                const boxes = JSON.parse(pyodide.runPython(`
                processor = PostProcessor(min_box_score=0.5, max_candidates=1000)
                arr = json.loads('${resultStr}')
                batch_boxes, batch_scores = processor(np.asarray(arr, dtype=np.float32), [(${img.height}, ${img.width})])
                boxes = order_boxes4nom(batch_boxes[0])
                json.dumps([box.tolist() for box in boxes])
                `));

                if(boxes.length > 0) {
                    const num2char = vocabulary();
                    const raw_image = await raw_tensor.array();
                    
                    // After we have (X1, Y1, X2, Y2, X3, Y3, X4, Y4) positions on the image
                    // We need to crop the image to a patch by positions
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

                    if(patches.length > 24) {
                        patches.pop();
                    }

                    if(patches.length <= 0) {
                        alert("[Detection model] can't find any box");
                        return;
                    }

                    predictBody.innerHTML = "";

                    // Run a for loop and process all patches
                    let pid = 0;
                    for(const patch of patches) {
                        // Build an element for HTML
                        const curPid = pid;
                        const col = document.createElement('div');
                        const colCard = document.createElement('div');
                        const cardImgWrapper = document.createElement('div');
                        const cardImg = document.createElement('img');
                        const colCardHeader = document.createElement('div')

                        col.classList.add('col-12', 'col-xl-4', 'col-lg-4');
                        colCard.classList.add('card', 'card-box', 'mb-3');
                        colCardHeader.classList.add('card-header');
                        cardImgWrapper.classList.add('card-img-top', 'loading')
                        cardImg.src = "assets/images/loading.gif";
                        cardImg.id = 'patch_' + (pid + 1);
                        colCardHeader.innerHTML = '<h3>Predicting...</h3>'

                        cardImgWrapper.appendChild(cardImg)
                        colCard.appendChild(cardImgWrapper);
                        colCard.appendChild(colCardHeader);
                        col.appendChild(colCard);

                        predictBody.appendChild(col);
                        
                        // Convert the patch to a Tensor (height, width, 3)
                        // And use tf.expandDims() to convert the Tensor to a shape (1, height, width, 3)
                        // The Recognition result is an array of tokens, we need to use CTC Greedy Decoder to convert it to array index list
                        // Ref: https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder
                        setTimeout(async () => {
                            const patchTensor = tf.tensor(patch, undefined, "float32");
                            const recResult = await CRNNModel.executeAsync({image: patchTensor.expandDims()});
                            const pred_tokens = await recResult.array();
                            const decoded = await ctcGreedyDecoder(pred_tokens[0]);
                            
                            // After we have an array index list, we get the text by each index number
                            let texts = await tokens2texts(decoded, num2char);
                            texts = texts.filter(x => x !== 'undefined').join('');
                            colCardHeader.innerHTML = `<h3>${texts}</h3>`

                            // Convert the patch to base64 image
                            const img = pyodide.runPython(`
                            from js import Blob, document
                            from js import window
                            import base64

                            box = get_patch(raw_image, boxes[${curPid}])
                            box = distortion_free_resize(box, align_top=True)            
                            success, encoded_im = cv2.imencode('.jpg', box)
                            encoded_str = base64.b64encode(encoded_im).decode('utf-8')
                            data_uri = f"data:image/jpeg;base64,{encoded_str}"
                            data_uri   
                            `)           
                            
                            cardImgWrapper.classList.remove('loading');
                            cardImg.src = img;

                            if(curPid === patches.length - 1) {
                                startButton.classList.remove('disabled');
                                startButton.removeAttribute('disabled');   
                                loading.classList.remove('active');                                  
                            }
                        }, 1000);

                        pid++;
                    }                     
                }
            }
        };
        
    });    

    loading.classList.remove('active');
})();