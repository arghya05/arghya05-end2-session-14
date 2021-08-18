# 14 BERT & BART

## Assignment

1.  TASK 1: Train BERT using the code mentioned  [here (Links to an external site.)](https://drive.google.com/file/d/1Zp2_Uka8oGDYsSe5ELk-xz6wIX8OIkB7/view?usp=sharing)  on the Squad Dataset for 20% overall samples (1/5 Epochs). Show results on 5 samples.
2.  TASK 2: Reproductive  [these (Links to an external site.)](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)  results, and show output on 5 samples.
3.  TASK 3: Reproduce the training explained in this  [blog (Links to an external site.)](https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c). You can decide to pick fewer datasets.
4.  Proceed to Session 14 - Assignment Solutions page and:
    1.  Submit README link for Task 1 (training log snippets and 5 sample results along with BERT description must be available) - 750
    2.  Submit README link for Task 2 (training log snippets and 5 sample results) - 250
    3.  Submit README link for Task 3 (training log snippets and 5 sample results along with BART description must be available) - 1000

## Solution



### Task 1 Results

BERT QA Bot on SQUAD Dataset

**Training Logs**

```
Train loss: 1.682131371960044
Saving model checkpoint to checkpoint-1000
```

![bert qa model training loss](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/14_BERT_BART/assets/bert_training.png?raw=true)

**Model Evaluation**

```
{
  "exact": 59.230009871668315,
  "f1": 61.210163805868284,
  "total": 6078,
  "HasAns_exact": 45.49828178694158,
  "HasAns_f1": 49.634149694868604,
  "HasAns_total": 2910,
  "NoAns_exact": 71.84343434343434,
  "NoAns_f1": 71.84343434343434,
  "NoAns_total": 3168,
  "best_exact": 59.493254359986835,
  "best_exact_thresh": -0.10016250610351562,
  "best_f1": 61.38637161248114,
  "best_f1_thresh": -0.08133554458618164
}
```

**Sample Outputs**

```
question       >> How does HT strive to give up power?
model's answer >> through "ideological struggle

question       >> When did Kublai ban the international Mongol slave trade?
model's answer >> 1291

question       >> What is the mayor of Warsaw called?
model's answer >> President

question       >> In what geographical portion of England is Abercynon located?
model's answer >> south Wales


```

### Task 2 Results

BERT Sentence Classification

**Training Logs**

```
======== Epoch 1 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:14.
  Batch    80  of    241.    Elapsed: 0:00:28.
  Batch   120  of    241.    Elapsed: 0:00:43.
  Batch   160  of    241.    Elapsed: 0:00:57.
  Batch   200  of    241.    Elapsed: 0:01:12.
  Batch   240  of    241.    Elapsed: 0:01:27.

  Average training loss: 0.40
  Training epcoh took: 0:01:27

Running Validation...
  Accuracy: 0.81
  Validation took: 0:00:04

======== Epoch 2 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:15.
  Batch    80  of    241.    Elapsed: 0:00:30.
  Batch   120  of    241.    Elapsed: 0:00:45.
  Batch   160  of    241.    Elapsed: 0:01:00.
  Batch   200  of    241.    Elapsed: 0:01:15.
  Batch   240  of    241.    Elapsed: 0:01:30.

  Average training loss: 0.27
  Training epcoh took: 0:01:30

Running Validation...
  Accuracy: 0.83
  Validation took: 0:00:04

======== Epoch 3 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:15.
  Batch    80  of    241.    Elapsed: 0:00:31.
  Batch   120  of    241.    Elapsed: 0:00:46.
  Batch   160  of    241.    Elapsed: 0:01:02.
  Batch   200  of    241.    Elapsed: 0:01:17.
  Batch   240  of    241.    Elapsed: 0:01:32.

  Average training loss: 0.18
  Training epcoh took: 0:01:33

Running Validation...
  Accuracy: 0.82
  Validation took: 0:00:04

======== Epoch 4 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:15.
  Batch    80  of    241.    Elapsed: 0:00:31.
  Batch   120  of    241.    Elapsed: 0:00:47.
  Batch   160  of    241.    Elapsed: 0:01:02.
  Batch   200  of    241.    Elapsed: 0:01:18.
  Batch   240  of    241.    Elapsed: 0:01:33.

  Average training loss: 0.13
  Training epcoh took: 0:01:34

Running Validation...
  Accuracy: 0.82
  Validation took: 0:00:04

Training complete!
```

![bert classification model training loss plot](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/14_BERT_BART/assets/bert-classification-training.png?raw=true)

**Sample Outputs**

```
sentence  > the table was wiped by john clean .
predicted < unacceptable
true cls  = unacceptable

sentence  > the book surprised many people .
predicted < acceptable
true cls  = acceptable

sentence  > of whom are you thinking ?
predicted < acceptable
true cls  = acceptable


```

### Task 3 Results

Training Logs: [https://wandb.ai/satyajit_meow/Paraphrasing%20with%20BART?workspace=user-satyajit_meow](https://wandb.ai/satyajit_meow/Paraphrasing%20with%20BART?workspace=user-satyajit_meow)

BART Paraphrasing

```
Text  > How can I be less insecure about my short height?
Pred  < How can I be less insecure about my height?
Truth = How can I be less insecure about my short height and looks?

Text  > What are tips for saving money?
Pred  < What are some tips for saving money?
Truth = What are the best methods of saving money?

Text  > Why November 14, birthday of Jawaharlal is celebrated as Children's day? What exceptional thing did he do to be considered as dearest to children?
Pred  < Why is the birthday of Jawaharlal Nehru celebrated as Children's Day? What exceptional thing did he do to be considered as dearest to children?
Truth = Why we celebrate Nehru's birthday as children's day?


```

## BERT

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers

BERT is basically a trained Transformer Encoder stack. Both BERT models (Large & Base) have large number of encoder layers (Transformer Blocks) - 12 for the Base version and 24 for the large version.

Model Input: The first input token is `[CLS]`, which stands for Classification. Just like a normal Transformer, BERT takes a sequence of words as input.
Model Outputs: Each position of the sequence outputs a vector of size `hidden_size`. For sentence classification we only focus on the first position (the `[CLS]` token position. The vector can now be used to classify into the class you chose. If you have more classes, this last layer (Classifier Network) is only changed.

As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional.

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a `[MASK]` token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence.

The BERT loss function takes into consideration only the prediction of the masked values and ignores the prediction of the non-masked words. As a consequence, the model converges slower than directional models.

In the BERT training process, the model receives pairs of sentences as input and learns to predict if the second sentence in the pair is the subsequent sentence in the original document. During training, 50% of the inputs are a pair in which the second sentence is the subsequent sentence in the original document, while in the other 50% a random sentence from the corpus is chosen as the second sentence. The assumption is that the random sentence will be disconnected from the first sentence.

## BART

**B**idirectional and **A**uto-**R**egressive **T**ransformers

BART is a denoising autoencoder built with a sequence-to-sequence model that is applicable to a very wide range of end tasks.

### Pretraining: Fill In the Span

BART is trained on tasks where spans of text are replaced by masked tokens, and the model must learn to reconstruct the original document from this altered span of text.

BART improves on BERT by replacing the BERT's fill-in-the-blank cloze task with a more complicated mix of pretraining tasks.

![text infilling](https://github.com/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/14_BERT_BART/assets/text_infilling.png?raw=true)

In the above example the origin text is ` A B C D E` and the span `C, D` is masked before sending it to the encoder, also an extra mask is placed between `A` and `B` and one mask is removed between `B` and `E`, now the corrupted document is `A _ B _ E`. The encoder takes this as input, encodes it and throws it to the decoder.

The decoder must now use this encoding to reconstruct the original document. `A B C D E`

---


