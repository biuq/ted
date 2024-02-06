let model = undefined;
let generationDelayInMilliseconds = 50;
let generationIntervalId = undefined;
let lastGeneratedText = "";
let genConfig = undefined;
let initialized = false;
let tokensProcessed = 0;
let contextSizeSum = [];
let weights = undefined;
let isLoading = false;

function start() {
    if (model === undefined) {
        return;
    }

    if (initialized === false) {
        reset();
    }

    if (generationIntervalId !== undefined) {
        clearInterval(generationIntervalId);
        generationIntervalId = undefined;
    }
    generationIntervalId = setInterval(generate, generationDelayInMilliseconds);
}

function stop() {
    if (model === undefined) {
        return;
    }

    if (generationIntervalId !== undefined) {
        clearInterval(generationIntervalId);
        generationIntervalId = undefined;
    }
}

function step() {
    if (initialized === false) {
        reset();
    }
    stop();
    generate();
}

function reset() {
    if (model === undefined) {
        return;
    }
    stop();
    model.reset();
    tokensProcessed = 0;
    contextSizeSum = new Array(model.config.num_layers).fill(0);
    processPrompt();
    initialized = true;
}

function processPrompt() {
    applyConfig();
    const prompt = document.getElementById('prompt');
    const text = [...prompt.value];
    lastGeneratedText = "";
    for (let x of text) {
        runModel(model, x);
    }
    output.textContent = prompt.value + lastGeneratedText;
    scroll();
}

function runModel(model, x) {
    applyConfig();
    lastGeneratedText = model(x);
    tokensProcessed++;

    for (let i = 0; i < model.layers.length; i++) {
        const contextSize = model.layers[i].getContext().entries.length;
        contextSizeSum[i] += contextSize;
    
        updateRow(i, i, contextSize, contextSizeSum[i] / tokensProcessed);
    }
}

function applyConfig() {
    const topK = getTopKElement().value;
    const topP = getTopPElement().value;
    const temperature = getTemperatureElement().value;
    const keepTopK = getKeepTopKElement().value;
    const threshold = getRelevanceScore().value;

    model.genConfig.top_k = topK;
    model.genConfig.top_p = topP;
    model.genConfig.temperature = temperature;
    model.updateContextConfig(keepTopK, threshold);
}

function generate() {
    if (model === undefined || initialized === false) {
        return;
    }

    const output = document.getElementById('output');
    runModel(model, [...lastGeneratedText][0]);
    output.textContent += lastGeneratedText;
    scroll();
}

function scroll() {
    const wrapper = document.getElementById("output-wrapper");
    wrapper.scrollBy(0, 100);
}

function getTopKElement() {
    return document.getElementById('topK');
}

function getTopPElement() {
    return document.getElementById('topP');
}

function getTemperatureElement() {
    return document.getElementById('temperature');
}

function getDelayElement() {
    return document.getElementById('delay');
}

function getKeepTopKElement() {
    return document.getElementById('keepTopK');
}

function getRelevanceScore() {
    return document.getElementById('relevanceScore');
}

function enforceMinMax() {
    var min = parseInt(this.min);
    var max = parseInt(this.max);
    if (parseInt(this.value) < min) {
        this.value = min;
    } else if (parseInt(this.value) > max) {
        this.value = max;
    }
}

getTopKElement().onchange = function(e) {
    enforceMinMax.apply(this);
};
getTopPElement().onchange = function(e) {
    enforceMinMax.apply(this);
};
getTemperatureElement().onchange = function(e) {
    enforceMinMax.apply(this);
};
getDelayElement().onchange = function(e) {
    enforceMinMax.apply(this);
    generationDelayInMilliseconds = this.value;
    if (generationIntervalId === undefined) {
        return;
    }
    stop();
    start();
};

getKeepTopKElement().onchange = function(e) {
    enforceMinMax.apply(this);
};

getRelevanceScore().onchange = function(e) {
    enforceMinMax.apply(this);
    updateVocabLifetimeList();
};

function lowestLayerPredictVocabLifetime() {
    const threshold = getRelevanceScore().value;
    const outProjWeight = loadTensor("out_proj.weight", weights);
    const embeddings = embeddingModule(outProjWeight);
    const layer = tedLayer(weights, { ...model.contextConfig, relevance_score: threshold }, model.config.norm_eps, 0);
    const chars = Object.keys(model.vocab.char_to_int);
    const lifetime = {};

    for (let i = 0; i < chars.length; i++) {
        layer.reset();
        x = embeddings(model.vocab.char_to_int[chars[i]]);
        layer(x);
        const entry = layer.getContext().entries[0];
        const magnitude = entry.magnitude;
        lifetime[chars[i]] = Math.log(threshold / magnitude) / entry.negativeLambda;
    }
    return lifetime;
}

function addRow(layerIndex, contextSize, avgContextSize) {
    const table = document.getElementById('table-body');

    const newRow = table.insertRow();
    const cell1 = newRow.insertCell(0);
    const cell2 = newRow.insertCell(1);
    const cell3 = newRow.insertCell(2);

    cell1.textContent = layerIndex;
    cell2.textContent = contextSize;
    cell3.textContent = parseFloat(avgContextSize).toFixed(2);
}

function updateRow(rowIndex, layerIndex, contextSize, avgContextSize) {
    const table = document.getElementById('table-body');

    if (rowIndex < 0 || rowIndex >= table.rows.length) {
        return;
    }

    const row = table.rows[rowIndex];
    row.cells[0].textContent = layerIndex;
    row.cells[1].textContent = contextSize;
    row.cells[2].textContent = parseFloat(avgContextSize).toFixed(2);
}

async function loadTinyStoriesModel() {
    if (model !== undefined) {
        return;
    }

    if (isLoading === true) {
        return;
    }

    isLoading = true;

    const model_and_weights = await loadModel('tiny_stories', (i,n) => {
        const progressBar = document.getElementById('progressBar');
        progressBar.value = (i / n) * 100;
    });
    model = model_and_weights[0];
    weights = model_and_weights[1];

    const topKElement = getTopKElement();
    topKElement.max = model.vocab.int_to_char.length;
    topKElement.value = model.genConfig.top_k;
    const topPElement = getTopPElement();
    topPElement.value = model.genConfig.top_p;
    const temperatureElement  = getTemperatureElement();
    temperatureElement.value = model.genConfig.temperature;
    const delayElement = getDelayElement();
    delayElement.value = generationDelayInMilliseconds;
    const keepTopKElement = getKeepTopKElement();
    keepTopKElement.value = model.contextConfig.top_k;
    const relevanceScoreElement = getRelevanceScore();
    relevanceScoreElement.value = model.contextConfig.relevance_score;

    for (let i = 0; i < model.config.num_layers; i++) {
        addRow(i, 0, 0);
    }

    updateVocabLifetimeList();

    isLoading = false;
}

function updateVocabLifetimeList() {
    const lifetime = lowestLayerPredictVocabLifetime();
    const listContainer = document.getElementById('vocabLifetimeList');
    listContainer.innerHTML = '';
    for (let x in lifetime) {
        const newItem = document.createElement('div');
        const value = lifetime[x];
        if (x === '\n') {
            x = '\\n';
        } else if (x === ' ') {
            x = '\\s';
        } else {
            x = `${x} `;
        }

        newItem.textContent = `${x}: ${value.toFixed(0)}`;
        listContainer.appendChild(newItem);
    }
}

async function loadModel(modelName, handleProgress) {
    const urls = getUrls(modelName)
    const [model,vocab,context,gen,weights] = await loadModelData(urls, handleProgress);
    const outProjNormWeight = loadTensor("out_proj_norm.weight", weights);
    const outProjWeight = loadTensor("out_proj.weight", weights);
    const embeddings = embeddingModule(outProjWeight);
    const outProjNorm = rmsNormModule(outProjNormWeight, model.norm_eps);
    const outProj = linearModule(outProjWeight);

    const layers = [];

    for (let i = 0; i < model.num_layers; i++) {
        layers.push(tedLayer(weights, context, model.norm_eps, i));
    }

    const fn = function(x) {
        x = embeddings(encode(vocab, [...x]));
        for (let layer of fn.layers) {
            x = layer(x);
        }

        const logits = outProj(outProjNorm(x));
        const sampled = sample(logits, fn.genConfig.top_k, fn.genConfig.top_p, fn.genConfig.temperature);
        return decode(vocab, [sampled]);
    };

    fn.reset = function() {
        for (let layer of fn.layers) {
            layer.reset();
        }
    };
    
    fn.updateContextConfig = function(topK, threshold) {
        for (let layer of fn.layers) {
            layer.updateContextConfig(topK, threshold);
        }
    }

    fn.config = model;
    fn.vocab = vocab;
    fn.contextConfig = context;
    fn.genConfig = gen;
    fn.layers = layers;

    return [fn, weights];
}

function loadModelData(urls, weightsDownloadProgressHandler) {
    const model_promise = loadJson(urls.model);
    const vocab_promise = loadJson(urls.vocab);
    const context_promise = loadJson(urls.context);
    const gen_promise = loadJson(urls.gen);
    const weights_promise = loadJsonWithProgress(urls.weights, weightsDownloadProgressHandler);

    return Promise.all([
        model_promise,
        vocab_promise,
        context_promise,
        gen_promise,
        weights_promise
    ]);
}

function getUrls(model_name) {
    return {
        model: `${model_name}/model.json`,
        vocab: `${model_name}/vocab.json`,
        context: `${model_name}/context.json`,
        gen: `${model_name}/gen.json`,
        weights: `${model_name}/weights.json`
    }
}

async function loadJson(url) {
    const response = await fetch(url);
    return await response.json();
}

async function loadJsonWithProgress(url, progressHandler) {
    const response = await fetch(url);
    const reader = response.body.getReader();
    const contentLength = +response.headers.get('Content-Length');
    
    let receivedLength = 0;
    let chunks = [];

    while(true) {
        const {done, value} = await reader.read();
        if (done) {
            break;
        }

        chunks.push(value);
        receivedLength += value.length;

        progressHandler(receivedLength, contentLength);
    }

    let chunksAll = new Uint8Array(receivedLength);
    let position = 0;

    for(let chunk of chunks) {
        chunksAll.set(chunk, position);
        position += chunk.length;
    }

    let result = new TextDecoder("utf-8").decode(chunksAll);
    let json = JSON.parse(result);
    return json;
}

function loadTensor(name, weights) {
    const weight = weights[name];
    const binary_string = window.atob(weight.data);
    const len = binary_string.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }
    const array = new Float32Array(bytes.buffer);    
    let stride = Array(weight.size.length);
    stride[weight.size.length - 1] = 1;
    for (let i = weight.size.length - 2; i >= 0; i--) {
        stride[i] = weight.size[i + 1] * stride[i + 1];
    }

    return {
        stride: stride,
        size: weight.size,
        numel: stride[0] * weight.size[0],
        data: array
    }
}

function embeddingModule(embeddings) {
    return function(idx) {
        let y = new Float32Array(embeddings.size[1]);
        for (let i = 0; i < y.length; i++) {
            y[i] = embeddings.data[idx * embeddings.stride[0] + i];
        }
        return y;
    }
}

function tedLayer(weights, contextConfig, normEps, idx) {
    const decayNormWeight = loadTensor(`layers.${idx}.decay_norm.weight`, weights);
    const lambdaMatrixWeight = loadTensor(`layers.${idx}.decay.lambda_matrix.weight`, weights);
    const quantityMatrixWeight = loadTensor(`layers.${idx}.decay.quantity_matrix.weight`, weights);
    const gateMatrixWeight = loadTensor(`layers.${idx}.decay.gate_matrix.weight`, weights);
    const outputMatrixWeight = loadTensor(`layers.${idx}.decay.output_matrix.weight`, weights);

    const ffnNormWeight = loadTensor(`layers.${idx}.ffn_norm.weight`, weights);
    const toHiddenWeight = loadTensor(`layers.${idx}.ffn.to_hidden.weight`, weights);
    const toHiddenGateWeight = loadTensor(`layers.${idx}.ffn.to_hidden_gate.weight`, weights);
    const toDimWeight = loadTensor(`layers.${idx}.ffn.to_dim.weight`, weights);

    const decayNorm = rmsNormModule(decayNormWeight, normEps);
    const decay = exponentialDecayModule(
        lambdaMatrixWeight, 
        quantityMatrixWeight, 
        outputMatrixWeight,
        gateMatrixWeight, 
        contextConfig
    );
    const ffnNorm = rmsNormModule(ffnNormWeight, normEps);
    const ffn = feedForwardModule(toHiddenWeight, toHiddenGateWeight, toDimWeight);
    const fn = function(x) {
        x = add(x, decay(decayNorm(x)));
        x = add(x, ffn(ffnNorm(x)));
        return x;
    };

    fn.reset = function() {
        decay.reset();
    };

    fn.updateContextConfig = function(topK, threshold) {
        decay.updateContextConfig(topK, threshold);
    };

    fn.getContext = function() {
        return decay.context;
    };

    return fn;
}

function exponentialDecayModule(
    lambdaMatrixWeight,
    quantityMatrixWeight,
    outputMatrixWeight,
    gateMatrixWeight,
    contextConfig,
) {
    const lambdaMatrix = linearModule(lambdaMatrixWeight);
    const quantityMatrix = linearModule(quantityMatrixWeight);
    const outputMatrix = linearModule(outputMatrixWeight);
    const gateMatrix = linearModule(gateMatrixWeight);
    const context = exponentialDecayContextModule(contextConfig);

    const fn = function(x) {
        const negativeLambda = logSigmoid(lambdaMatrix(x));
        const quantity = quantityMatrix(x);

        context.tick();
        const output = context.influence(quantity);
        context.add(negativeLambda[0], quantity);
        context.runEviction();

        return mul(outputMatrix(silu(output)), gateMatrix(x));
    };

    fn.reset = function() {
        fn.context.reset();
    };

    fn.updateContextConfig = function(topK, threshold) {
        fn.context.contextConfig.top_k = topK;
        fn.context.contextConfig.relevance_score = threshold;
    };

    fn.context = context;

    return fn;
}

function exponentialDecayContextModule(contextConfig) {
    return {
        entries: [],
        contextConfig,
        add(negativeLambda, quantity) {
            this.entries.push(exponentialDecayContextEntry(negativeLambda, quantity));
        },
        tick() {
            for (let entry of this.entries) {
                entry.time += 1; 
            }
        },
        runEviction() {
            this.keepTopK(this.contextConfig.top_k, contextConfig.min_tokens_to_keep);
            this.keepAboveThreshold(this.contextConfig.relevance_score, contextConfig.min_tokens_to_keep);
        },
        keepTopK(topK, minTokensToKeep) {
            if (topK > 0 && this.entries.length > 0) {
                topK = Math.max(topK, minTokensToKeep);
                topK = Math.min(topK, this.entries.length);
                this.entries.sort((a, b) => a.relevance_score() - b.relevance_score());
                this.entries = this.entries.slice(-topK);
            }
        },
        keepAboveThreshold(threshold, minTokensToKeep) {
            const alive = [];
            
            for (let entry of this.entries) {
                if (entry.relevance_score() >= threshold) {
                    alive.push(entry);
                }
            }

            if (alive.length < minTokensToKeep){
                this.keepTopK(minTokensToKeep, minTokensToKeep);
            } else {
                this.entries = alive;
            }
        },
        influence(token) {
            const y = [...token];
            for (let entry of this.entries) {
                const influence = entry.influence();
                for (let i = 0; i < token.length; i++) {
                    y[i] += influence[i];
                }
            }
            return y;
        },
        reset() {
            this.entries = [];
        }
    }
}

function exponentialDecayContextEntry(negativeLambda, quantity) {
    return {
        time: 0,
        negativeLambda,
        quantity,
        magnitude: magnitude(quantity),
        decay_factor() {
            return Math.exp(this.negativeLambda * this.time);
        },
        influence() {
            return mulByScalar(this.quantity, this.decay_factor());
        },
        relevance_score() {
            return this.decay_factor() * this.magnitude;
        }
    }
}

function feedForwardModule(toHiddenWeight, toHiddenGateWeight, toDimWeight) {
    const toHidden = linearModule(toHiddenWeight);
    const toHiddenGate = linearModule(toHiddenGateWeight);
    const toDim = linearModule(toDimWeight);
    return function(x) {
        const h = toHidden(x);
        const g = toHiddenGate(x);
        return toDim(mul(h, silu(g)));
    }
}

function sample(logits, topK, topP, temperature, min_tokens_to_keep=1, filter_value=-Infinity) {
    logits = mulByScalar(logits, 1 / temperature);
    if (topK > 0) {
        topK = Math.min(topK, logits.length);
        let topk_values = [...logits].sort((a, b) => a - b).slice(-topK);
        let min = +Infinity;
        for (let i = 0; i < topK; i++) {
            min = Math.min(topk_values[i], min);
        }
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] < min) {
                logits[i] = filter_value;
            }
        }
    }

    if (0 <= topP && topP <= 1.0) {
        const sorted = logits.map((v, i) => ({v, i})).sort((a, b) => a.v - b.v);
        const sorted_logits = sorted.map(x => x.v);
        const sorted_indices = sorted.map(x => x.i);
        const cum_probs = cumsum(softmax(sorted_logits));
        let removed = 0;

        for (let i = 0; i < cum_probs.length; i++) {
            if (cum_probs[i] <= (1 - topP) && (cum_probs.length - removed) > min_tokens_to_keep) {
                logits[sorted_indices[i]] = filter_value;
                removed++;
            }
        }
    }

    const uniform_sample = Math.random();
    const cdf = cumsum(softmax(logits));
    let sample = 0;
    for (let i = 0; i < cdf.length; i++) {
        if (uniform_sample <= cdf[i]) {
            sample = i;
            break;
        }
    }

    return sample;
}

function cumsum(x) {
    const y = new Float32Array(x.length);
    let sum = x[0];
    y[0] = sum;
    for (let i = 1; i < x.length; i++) {
        sum += x[i];
        y[i] = sum;
    }
    return y;
}

function softmax(x) {
    let max = -Infinity;
    for (let i = 0; i < x.length; i++) {
        max = Math.max(max, x[i]);
    }
    x = [...x];
    for (let i = 0; i < x.length; i++) {
        x[i] = x[i] - max;
    }
    let y = new Float32Array(x.length);
    let sum = 0.0;
    for (let i = 0; i < x.length; i++) {
        const exp = Math.exp(x[i]);
        sum += exp;
        y[i] = exp;
    }

    for (let i = 0; i < y.length; i++) {
        y[i] = y[i] / sum;
    }

    return y;
}

function magnitude(x) {
    let sum = 0.0;
    for (let i = 0; i < x.length; i++) {
        sum += x[i] * x[i];
    }
    return Math.sqrt(sum);
}

function logSigmoid(x) {
    let y = new Float32Array(x.length);
    for (let i = 0; i < y.length; i++) {
        y[i] = Math.log(1 / (1 + Math.exp(-x[i])));
    }
    return y;
}

function sigmoid(x) {
    let y = new Float32Array(x.length);
    for (let i = 0; i < y.length; i++) {
        y[i] = 1 / (1 + Math.exp(-x[i]));
    }
    return y;
}

function silu(x) {
    let y = new Float32Array(x.length);
    for (let i = 0; i < y.length; i++) {
        y[i] = x[i] / (1 + Math.exp(-x[i]));
    }
    return y;
}

function linearModule(weight) {
    return function(x) {
        const [out_features, in_features] = weight.size;
        const [out_stride, _] = weight.stride;
        const y = new Float32Array(out_features);

        for (let out_i = 0; out_i < out_features; out_i++) {
            let sum = 0.0;
            for (let in_i = 0; in_i < in_features; in_i++) {
                sum += x[in_i] * weight.data[out_i * out_stride + in_i];
            }
            y[out_i] = sum;
        }
        return y;
    }
}

function rmsNormModule(weight, eps) {
    return function(x) {
        const square = powToScalar(x, 2);
        let sum = 0;
        for (let i = 0; i < x.length; i++) {
            sum += square[i];
        }
        const mean = sum / x.length;
        const norm = 1 / Math.sqrt(mean + eps);
        
        return mul(mulByScalar(x, norm), weight.data);
    }
}

function encode(vocab, text) {
    const y = [];
    for (let i = 0; i < text.length; i++) {
        y.push(vocab.char_to_int[text[i]]);
    }
    return y;
}

function decode(vocab, token_ids) {
    let y = "";
    for (let i = 0; i < token_ids.length; i++) {
        y += vocab.int_to_char[token_ids[i]];
    }
    return y;
}

function powToScalar(x, scalarExponent) {
    let y = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
        y[i] = Math.pow(x[i], scalarExponent);
    }
    return y;
}

function mulByScalar(x, scalar) {
    let y = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
        y[i] = x[i] * scalar;
    }
    return y;
}

function mul(a, b) {
    let y = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
        y[i] = a[i] * b[i];
    }
    return y; 
}

function add(a, b) {
    let y = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
        y[i] = a[i] + b[i];
    }
    return y; 
}
