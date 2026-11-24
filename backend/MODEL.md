# The sign recognition model (LSTM)

This explains what `train.py` builds and why, layer by layer, so you can research each part and write about it. Code: `build_model()` in [train.py](train.py).

## 1. The problem the model solves

A sign is a movement, not a pose. HELLO and BYE-BYE can pass through similar hand shapes; what tells them apart is how the hands move over time. So the input is a **sequence**: 60 frames, and in each frame 126 numbers (2 hands × 21 landmarks × x, y, z), prepared by `preprocess.py`. The output is one probability for each of the 10 signs.

```
recording  →  preprocess.py  →  (60 × 126)  →  LSTM model  →  [0.02, 0.85, 0.01, …]  →  "FATHER", 85%
```

This is **sequence classification**: many time steps in, one label out.

## 2. Why an LSTM

A normal fully connected network takes a fixed list of numbers and has no idea that frame 12 comes after frame 11. A **recurrent neural network (RNN)** reads the frames one at a time and carries a *hidden state* (its memory) from one frame to the next, so what it "thinks" at frame 30 depends on frames 1–29.

Plain RNNs struggle to remember things for more than a few steps: during training the error signal shrinks (or explodes) as it is passed back through many time steps, the *vanishing gradient problem* (Bengio et al., 1994). The **Long Short-Term Memory** (LSTM) cell (Hochreiter & Schmidhuber, 1997) fixes this with a separate *cell state* and three **gates** that learn what to keep, what to add and what to output:

| Gate | Question it answers | In a sign |
|---|---|---|
| Forget gate *f* | What from the memory is no longer useful? | Drop the starting position once the movement has begun |
| Input gate *i* | What from this frame is worth storing? | "The hand just moved up to the forehead" |
| Output gate *o* | What part of the memory matters right now? | Pass on the movement pattern to the next layer |

The forget gate was added by Gers et al. (2000). The equations, for frame *t* with input *xₜ* and previous hidden state *hₜ₋₁* (σ = sigmoid, 0 to 1; ⊙ = element-wise multiply):

```
fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)        forget gate
iₜ = σ(W_i·[hₜ₋₁, xₜ] + b_i)        input gate
c̃ₜ = tanh(W_c·[hₜ₋₁, xₜ] + b_c)     candidate new memory
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ            cell state (long-term memory)
oₜ = σ(W_o·[hₜ₋₁, xₜ] + b_o)        output gate
hₜ = oₜ ⊙ tanh(cₜ)                  hidden state (what the cell outputs)
```

Because the cell state is updated by *adding* (`+ iₜ ⊙ c̃ₜ`) rather than repeatedly multiplying, the error signal survives across all 60 frames. Olah (2015) has the clearest illustrated explanation of these equations; start there.

**Why not something else?** (worth a paragraph in the report)
- *GRU* (Cho et al., 2014): a simpler cell with two gates; often similar accuracy, fewer parameters. A good comparison experiment.
- *1D CNN over time*: fast, looks at short windows of frames; also a sensible baseline.
- *Transformer*: strong on sequences but needs much more data than 12 videos per sign.
- *3D CNN on raw video*: rejected in A1 (compute, and video would have to leave the device).

An LSTM on landmarks is small, fast on a CPU, well documented, and a standard choice for isolated sign recognition from keypoints, which matches the A1 decisions.

## 3. The layers, one by one

```
Input (60, 126)
Masking(0)
LSTM(64, return_sequences=True)    48,896 parameters
Dropout(0.3)
LSTM(32)                           12,416
Dropout(0.3)
Dense(32, relu)                     1,056
Dense(10, softmax)                    330
                         total     62,698
```

### Input: `(60, 126)`
60 time steps because `preprocess.py` resamples every recording to 60 frames (the stage-1 videos have a median of 68 frames after trimming, so little is lost or invented). 126 features per step: left hand's 21 points × 3, then the right hand's. Keras adds a batch dimension, so the real shape is `(batch, 60, 126)`; the API sends a batch of one.

### `Masking(mask_value=0.0)`
When no hand is visible, `preprocess.py` writes zeros. A frame where **all 126 values are 0** (both hands missing) carries no information, and treating it as "both hands at the wrist position" would mislead the model. Masking tells the LSTM layers to skip those time steps and carry their memory over unchanged. A single missing hand is not masked (the other 63 numbers are real); the model learns that zeros in one half mean "this hand isn't there".

### `LSTM(64, return_sequences=True)`
The first recurrent layer: 64 LSTM units read the 60 frames in order. `return_sequences=True` makes it output its hidden state at **every** frame, `(60, 64)`, rather than only at the end, because the next LSTM needs a sequence to read.

Think of it as learning low-level movement features: "hand moving up", "fingers closing", "two hands meeting". 64 is a common size for small keypoint datasets: big enough to describe two hands' motion, small enough not to memorise 86 training videos.

Parameters: an LSTM has 4 weight sets (3 gates + candidate), each connecting inputs and the previous hidden state to every unit, plus a bias:
`4 × (64 × (126 + 64) + 64) = 48,896`.

### `Dropout(0.3)`
During training, randomly sets 30% of the 64 outputs to zero at each step, a different 30% every batch (Srivastava et al., 2014). The network can't rely on any single unit, so it learns more general patterns. This is the main defence against **overfitting**: with only 8–9 real videos per sign, a model can otherwise memorise them perfectly and fail on new people. Dropout is switched off automatically when predicting.

### `LSTM(32)`
The second recurrent layer reads the 64-number sequence and, without `return_sequences`, outputs only its final hidden state: **one vector of 32 numbers summarising the whole sign**. Stacking two LSTMs lets the second combine the first one's simple movements into whole-sign patterns ("hand goes up, then out, then closes"). It is smaller (32) because it summarises, and fewer parameters means less overfitting.
Parameters: `4 × (32 × (64 + 32) + 32) = 12,416`.

### `Dropout(0.3)`
Same idea, applied to the 32-number summary.

### `Dense(32, activation="relu")`
A fully connected layer: every one of its 32 units looks at all 32 summary numbers. It turns "what the movement looked like" into "which features point to which sign". ReLU (`max(0, x)`) is the standard activation: it lets the layer model non-linear combinations and trains well (Nair & Hinton, 2010).
Parameters: `32 × 32 + 32 = 1,056`.

### `Dense(10, activation="softmax")`
The output: one unit per sign. Softmax turns the 10 raw scores into probabilities between 0 and 1 that add up to 1. The highest one is the answer, and its value is the **confidence** the app shows. The next highest are the "top guesses".
Parameters: `32 × 10 + 10 = 330`.

Note that softmax always gives *some* answer: a random movement still gets a "best" sign. A low confidence (e.g. under 50%) is the app's hint that the model isn't sure. Rejecting unknown signs properly is future work.

## 4. How it learns

- **Loss: sparse categorical cross-entropy.** For each video, `−log(probability given to the correct sign)`. 90% on the right sign costs 0.11; 10% costs 2.3. "Sparse" just means labels are numbers (0–9) instead of one-hot vectors.
- **Optimiser: Adam, learning rate 0.001** (Kingma & Ba, 2015). Gradient descent with an individual, self-adjusting step size per weight; the standard default that works without much tuning.
- **Batch size 32**: weights are updated after every 32 videos.
- **Epoch**: one pass through all training videos (≈1,800 with augmentation).
- **Early stopping** (Prechelt, 1998): after each epoch the model is checked on the validation videos it does not train on. When validation loss hasn't improved for 30 epochs, training stops and the weights from the best epoch are restored. This picks the point before the model starts memorising.

## 5. Data: split and augmentation

**Split per sign** (`split_per_sign`): of each sign's 11–12 usable videos, 2 go to the **test set**, 1 to the **validation set**, the rest (8–9) to **training**. The test videos are never used for training or for choosing when to stop, so test accuracy estimates performance on unseen videos. Every sign gets the same number of test videos so no sign dominates the score.

**Augmentation** (`augment`): each training video gets 20 altered copies, made fresh from the original:

| Change | Range | Real-world variation it imitates |
|---|---|---|
| Speed / timing | cut up to 10% from each end, stretch to 60 frames | People sign faster or slower; recording starts a bit early/late |
| Rotation | ±15° in the image plane | Tilted camera or head, different posture |
| Scale | ±10% | Different hand proportions |
| Noise | Gaussian, σ = 0.02 | MediaPipe's landmarks jitter between frames |
| Mirror | 50% of copies: flip x, swap left/right hand | Left-handed signers (Auslan signers use their dominant hand) |

Zeros (missing hands) are kept as exact zeros so masking still works. Augmentation is applied only to training data; validation and test are real, unaltered videos. Data augmentation is well established for small datasets (Shorten & Khoshgoftaar, 2019, survey it for images; the same idea applies to landmark sequences).

Distance from the camera and position in the frame need no augmentation: `preprocess.py` already removes them (centred on the wrist, scaled by hand size).

## 6. Results (stage 1: 10 signs)

Test set: 20 videos (2 per sign). Different seeds give different random splits:

| Seed | Test accuracy | Top-3 accuracy | Epochs run (best epoch) |
|---|---|---|---|
| 42 (saved model) | 80% (16/20) | 95% | 38 (8) |
| 1 | 85% (17/20) | 100% | 49 (19) |
| 2 | 80% (16/20) | 95% | 55 (25) |
| 3 | 90% (18/20) | 100% | 41 (11) |
| 4 | 70% (14/20) | 85% | 102 (72) |
| **Mean** | **81%** (range 70–90%) | **95%** | |

Chance level with 10 signs is 10%. Top-3 means the right sign was among the three best guesses. In the seed-42 run, the errors were HELLO → BYE-BYE, TEACHER → BYE-BYE, HELP → FATHER (one of the two videos each); every other sign had both test videos right.

**Read these numbers carefully in the report:**
- 20 test videos means each video is 5 percentage points; one mistake moves the score a lot. Report the range over several seeds, not one number.
- Validation is only 10 videos, so early stopping is noisy (it sometimes picks a very early epoch).
- We don't know whether the same people appear in training and test videos. If they do, accuracy on new signers (your webcam) will be lower. Test with group members' webcam recordings and report that separately; it's the more honest measure for the project.
- Webcam recordings differ from the dataset (camera, lighting, angle, signing style). Expect a drop.

## 7. Things to try next (good for the A3 "iterations")

1. GRU instead of LSTM; compare accuracy and size.
2. Bidirectional LSTM (reads the sign forwards and backwards).
3. k-fold cross-validation for a more stable accuracy figure.
4. Add face/body landmarks (some signs are made at the forehead or chest: MOTHER, FATHER, THANK YOU).
5. More signers' data: group webcam recordings added to training.
6. A confidence threshold or "unknown" class.

## References

Check each one before citing (APA 7).

- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks, 5*(2), 157–166. https://doi.org/10.1109/72.279181
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing* (pp. 1724–1734). https://doi.org/10.3115/v1/D14-1179
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation, 12*(10), 2451–2471. https://doi.org/10.1162/089976600300015015
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org (chapter 10: sequence modelling, recurrent nets and LSTMs)
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations*. https://arxiv.org/abs/1412.6980
- Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. In *Proceedings of the 27th International Conference on Machine Learning* (pp. 807–814).
- Olah, C. (2015, August 27). *Understanding LSTM networks*. colah's blog. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Prechelt, L. (1998). Early stopping — but when? In G. B. Orr & K.-R. Müller (Eds.), *Neural networks: Tricks of the trade* (pp. 55–69). Springer. https://doi.org/10.1007/3-540-49430-8_3
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*, Article 60. https://doi.org/10.1186/s40537-019-0197-0
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958.
- Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). *MediaPipe Hands: On-device real-time hand tracking*. arXiv. https://arxiv.org/abs/2006.10214
