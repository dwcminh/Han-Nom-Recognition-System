class ConvBnRelu extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.filters = config.filters;
        this.kernelSize = config.kernelSize;
        this.padding = config.padding;
        this.useBias = config.useBias;
    }

    build(inputShape) {
        this.conv = tf.layers.conv2d({
            filters: this.filters,
            kernelSize: this.kernelSize,
            padding: this.padding,
            useBias: this.useBias,
            activation: 'linear', // No activation here, will apply BN and ReLU in `call`
        }); // Fixed to properly define and use the layer in `call`

        this.bn = tf.layers.batchNormalization(); // Correctly define for use in `call`
    }

    // Data is passed through Convolution layer
    // -> Batch Normalization layer -> ReLU activation function.
    call(inputs, training = false) {
        let x = this.conv(inputs); // Correctly use the convolution layer
        x = this.bn(x, training); // Correctly use batch normalization with the training flag
        return tf.relu(x); // Apply ReLU activation function
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            filters: this.filters,
            kernelSize: this.kernelSize,
            padding: this.padding,
            useBias: this.useBias,
        });
        return config;
    }

    static get className() {
        return 'ConvBnRelu';
    }
}

class DeConvMap extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.filters = config.filters;
    }

    build(inputShape) {
        this.convBnRelu = new ConvBnRelu({
            filters: this.filters,
            kernelSize: 3,
            useBias: false
        });
        this.deconv1 = tf.layers.conv2dTranspose({
            filters: this.filters,
            kernelSize: 2,
            strides: 2,
            useBias: false
        });
        this.bn = tf.layers.batchNormalization();
        this.relu = tf.relu();
        this.deconv2 = tf.layers.conv2dTranspose({
            filters: 1,
            kernelSize: 2,
            strides: 2,
            activation: 'sigmoid'
        });
    }

    // Data is passed through a series of layers to perform deconvolution and other transformations
    call(inputs, kwargs) {
        let x = this.convBnRelu(inputs);
        x = this.deconv1(x);
        x = this.bn(x);
        x = this.relu(x);
        x = this.deconv2(x);
        return tf.squeeze(x, [-1]); // Remove all dimensions of size 1
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            filters: this.filters
        });
        return config;
    }

    static get className() {
        return 'DeConvMap';
    }
}

class ApproximateBinaryMap extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.k = config.k; // The scaling factor
    }

    // Performs a logistic transformation between two inputs to produce a new output
    // Return tensor contains values in the range [0, 1]
    call(inputs, kwargs) {
        // Assumes inputs is an array of two tensors: [binarize_map, threshold_map]
        let binarize_map = inputs[0];
        let threshold_map = inputs[1];
        // Perform the operation: 1 / (1 + exp(-k * (binarize_map - threshold_map)))
        return tf.div(tf.scalar(1), tf.add(tf.scalar(1), tf.exp(tf.mul(tf.scalar(-this.k), tf.sub(binarize_map, threshold_map)))));
    }

    getConfig() {
        const config = super.getConfig();
        config.k = this.k;
        return config;
    }

    static get className() {
        return 'ApproximateBinaryMap';
    }
}


tf.serialization.registerClass(ConvBnRelu);
tf.serialization.registerClass(DeConvMap);
tf.serialization.registerClass(ApproximateBinaryMap);