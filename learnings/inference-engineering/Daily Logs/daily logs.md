
# Month 1 (from 2026-08-24 to 2026-09-18) : 26 days

Plan reference [file](../Curriculum/Month%201.md)

## 2026-08-24

Timespent: 1 hour

Installed UV and setup the .venv environment. 
Did a revision of autograd. Bit rusty, but I recalled what I learnt before.
Tomorrow learn the pytorch basics tutorial from https://docs.pytorch.org/tutorials/beginner/basics/intro.html

## 2026-08-30

Timespent: 1 hour
Finished pytorch tensors, datasets, transforms. Was able to recollect things.
Next learn about building Neural network, training and saving/loading the model. https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

## 2026-09-10

Timespent: 2.5 hours
Finished basic model building, autograd (mainly .detach()), optimizing a model. Understood the below

```
.detach()
model.train() model.eval()
torch.no_grad() outside the for loop
```

Next learn about storing and loading a model. Finish off blitz (it's same as fashionmnist but on diff dataset) and learn parallelism

## 2026-09-11

Timespent: 2 hours

Finished storing and loading a model. Learnt is how state_dict() is used, stored and be pushed back to another model object. CIFAR10 basic training loop and CNN model has been done. Skipped other parts in https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.

Next learn about parallelism in PyTorch https://docs.pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

## 2026-09-12

Timespent: 1.5 hours

Finished learning about data parallelism in PyTorch. Understood how to use `torch.nn.DataParallel` to distribute the model across multiple GPUs and how to handle inputs and outputs accordingly. Why x and y has to be in the same device as the model for proper parallel computation. Saw "Distributed Training in PyTorch: Zero to Hero - Corey Lowman, Lambda Labs" video. 

Next read https://einops.rocks/1-einops-basics/ and if possible, do https://einops.rocks/2-einops-for-deep-learning/ as well.

## 2026-09-16

Timespent: 1.25 hours

Finished einops basics and how it's used as a PyTorch layer. Started with basic pytest to hand derive gradients in torch and compare it with loss.backward(). Facing issue now. Need to check it in the next day. With that, see all the things we have seen and wrap up the month 1.

## 2026-09-17

Timespent: 30 mins

Just understood the neural network building with binary classifier part. Didn't do anything else today. Was so distracted of the office work, couldn't focus much on learning. Tomorrow will continue with the pytest writing and gradient checking.


## 2026-09-18

Timespent: 1.5 hours

Wrote the gradient finding with proper pytorch code and compared it to the grad generate by the `.backward()` and written assert codes to check the shape of my grad vs pytorch grad and it's value. Everything seems to be working correctly.

We can consider this month's session as closed.

In next session, start the [Month2.md](../Curriculum/Month2.md)'s Stage 0. Session 0 is already done, Session 1 is to properly add functions in the CIFAR10 model and see what can be done about session 2 as it involves running code in GPU.

# Month 2 (from 2026-09-19 to 2026-10-16) : 28 days

## 2026-09-19

Timespent: 4-5 scattered hours through this Saturday

Consider the session 0 is not a pending one. Almost finished session 1. Pending is storing and loading all proper things like otpimizer, scaler, epoch, config, model state_dict and just not the model state_dict alone. https://www.perplexity.ai/search/acc75eb6-ea0e-4fd2-bdf3-69068c32f51f

Finish off the above one in next session, add timer to each epoch and track it properly. Then go for session 2 in runpod.io to run the code on GPU.

## 2026-09-20

Timespent: 3-4 scattered hours till now in this Sunday

Finished session 1 with properly handling all the cases, separating the entire single file of code to proper packages and modules for better organization and maintainability.

Took diff performance stats of mps vs cuda in diff precision modes. So, calling session 2 as "complete". Tomorrow watch https://www.youtube.com/watch?v=LuhJEEJQgUM and try to answer the mentioned questions in curriculum

## 2026-09-21

Timespent: 1 hour

Just watched the GPU Mode lecture 1. Got some gist of things.

Tomorrow https://www.perplexity.ai/search/856f19e9-b550-4d06-9d75-b749ceb32584 in this thread, I need to answer/discuss the questions with AI, to know the things to be understood by me in this session. 

## 2026-09-23

Timespent: 2 hours

Rewatched the GPU Mode lecture 1 to get better understanding. In a new https://www.perplexity.ai/search/dab50fc5-d248-4439-bcfa-9e9ad0c5ff25 thread, I have cross checked my understanding of the questions mentioned in the curriculum. Let's mark session 4 as done.

For session 5 as per Month2 Session5, tomorrow finish off the prewatch and watch lecture 4


## 2026-09-24

Timespent: 1 hour

I have finished the prewatch for session 5 and watched lecture 4. Notes availalble in the [Month 2 Notes.md](../Notes/Month%202/Session%205.md).

Tomorrow in the https://www.perplexity.ai/search/578881e8-e8ba-4700-9ea9-95ae466be9e7 and do `!nvidia-smi` to check the GPU status. Finish session 5 and take notes accordingly.

## 2026-10-03

Didn't continue with before planned one. But to avoid touching the work. I just answered a question from the new perplexity link https://www.perplexity.ai/search/6a42688d-725b-4cfa-8639-b9e84605dac8 . 

In the next session, I will continue with session 5 as planned, check the GPU status using `!nvidia-smi`, and take notes accordingly.

## 2026-10-04

Time spent: 50 mins + 15 mins

Finished the perplexity link. I still not know about the internals of GPU 100%, but have a pretty good idea. I feel that it's ok as of now. Finished session 5 with `!nvidia-smi` and checkpoint questions and answers.

Tomorrow, go for session 6. First read what's there and what I have to do, to get a gist and proceed as much as I can.

## 2026-10-05

Timespent: 1 min

Just for chain of work, I just read fp32 and nothing else. Tomorrow I need to work as per the plan.

## 2026-10-06

Timespent: 1 hour (scattered)

started the pytorch amp tutorial and stopped with the autocast & gradscaler, check the slowness of autograd in cpu. Tomorrow, run the same in the GPU. Finish session 6 by tomorrow. I have to wakeup by 5AM rather than 6AM.

## 2026-10-07

Timespent: 30 mins

Finished session 6 with proper understanding of autocast, gradscaler, and their interactions on CPU and GPU. Created notes as well. Started with session 7 and did basic profiling with CPU & GPU.

Tomorrow I need to learn how the naming of sort-by is there from [this](https://www.perplexity.ai/search/2dec4c50-825d-4697-91e1-8cf965f040a3) and finish part a and part b of session 7.

## 2026-10-08

Timespent: 1.5 hours

Finished session 7 fully and updated notes properly. I should've wokeup at 5AM to have a non-disturbed morning for focused work. Tomorrow for session 8, I need to build a CNN from scratch to have 70% accuracy on CIFAR-10 dataset on fp32 and log the traces and everything. Let's see how much I can get it done. It's because I have to get the data loading properly, transformation, initital run properly.

## 2026-10-09

Timespent: 1 hour

I should wake up a bit sooner around 5:30 or something. From session 8, I finished data loading, transformation, model architecture and counted parameters. Tomorrow, I need to define loss, write & run the training loop, testing loop and set the fp32 baseline.

## 2026-10-10

Timespent: 1.5 hours

Finished session 8, for session 9 everything is same but we are implementing mixed precision training with fp16. I need to modify the training loop accordingly and check the performance and accuracy by Monday. Handle properly about the inititaing the model with same weights for reproducing the results and restoring the state_dict of gradscaler for resuming training.

## 2026-10-12

Timespent: 1 hour

Finished coding of session 9 and took output traces and compared it. Still some questions are pending. Start with checking https://www.perplexity.ai/search/0ffadfae-c23a-4b92-81ec-e4e5c001f56c tomorrow. check the kernel differences. Recheck the session 9 notes for the questions.

I guess I can finish both session 9 and 10 tomorrow.

## 2026-10-13

Timespent: 10 mins

Wokeup at 6AM, but kids got me occupied. So didn't do much. ALso, I was distracted with other work (which I shouldn't have started). Tomorrow start with https://www.perplexity.ai/search/0ffadfae-c23a-4b92-81ec-e4e5c001f56c and finish session 9 and hop to session 10.


## 2026-10-14

Timespent: 1 hour

Done with session 9, properly took the profiling traces for fp32 and amp and compared and written answers to the questions.

Finish session 10 tomorrow and see what is month 3 about.

## 2026-10-15

Timespent: 1 hour

In session 10, out of 15 questions, 10 are done https://www.perplexity.ai/search/2c42c5b9-8d7a-4893-a70e-2ea85da36fc1 . 

- [x] Learn fp32 vs fp16 vs bf16 https://www.perplexity.ai/search/4b7966dc-74c2-4a02-b055-5a09739d1164?preview=1
- [x] I need to finish balance 5 tomorrow
- [x] Write top to bottom lecture notes
- [x] See month3's work and wrap up the mind.

## 2026-10-16

Timespent: 1 hour but distracted with local area early morning music.

But I got the work done somehow. For Month3, finish session 0 to session 4 in 1 shot, as I have already finished micrograd before. So, read that code, understand it, internalize it again. Inform the same in notes properly. Then see the problem statement in session 5

# Month 3 (from 2026-10-17 to 2026-10-19) : 3 days

## 2026-10-17

Timespent: 2-3 hours scattered

Finished till session 4 and coded micrograd and did a sample training loop with loss being minimized for each epoch.

## 2026-10-18

Timespent: 4 hours scattered

Coded bigram model with plain python and single neuron and tested it. Main thing for me to remember is that we used the NN based model to compute the probability distribution over the next character given the current character, which we found directly in the count based model. Another gotcha is how float16 changed 4410 to 4408.

Tomorrow is holiday. So I can spend more time on session 8, which is building the makemore but with multi-layered perceptrons. Yeah.. Start with watching https://youtu.be/TCH_1BHY58I first to get good glimpse and for faster coding.

## 2026-10-19

Timespent: 3 hours scattered

Recalled MLP from Andrej's ipynb and coded it without seeing his implementation (except for 2 things). All good. We can wrap the month 3's work now and move on to the next part of the curriculum. From 17-19, I had 3 days holiday, so I was able to spend a good amount of time to get the month3 works done. Feeling satisfied with the progress. I need to write curriculum for month4 and get started from tomorrow. Created in-depth curriculum for Month4.

- [ ] start the batch normalization and understand it deeply


# Month 4 (from 2026-10-20) : In progress

## 2026-10-20

Timespent: 1.5 hours

Instead of watching https://youtu.be/P6sfmUTpUmc , I implemented the batchnorm and it's results are not as I expected. tomorrow I need to watch the video. Day after that, I can sit and finish off with the initialization experiments and then proceed for better loss minimization with batchnorm.

- [x] Watch https://youtu.be/P6sfmUTpUmc (1 hour 55 minutes)

## 2026-10-21

Timespent: 40mins

Wokeup late because of night travel and delayed sleep. Today, I finished the video till the initialization experiments part. Tomorrow, I'll see the remaining part of the video and finish it.

- [x] Watch https://youtu.be/P6sfmUTpUmc?t=2448

## 2026-10-22

Timespent: 1.5 hours

Finished watching the video. This marks the finish of session 1 of month4. Tomorrow I need to finish session 2 - Initialization experiments	Measured logits/activation/gradient effects.

## 2026-10-23

Timespent: 1.5 hours

Finished session 2 - Initialization experiments and documented in the notes4.md. Tomorrow session 3 with batch norm implementation and if possible write BatchNorm1D code. https://youtu.be/P6sfmUTpUmc?t=2428 (for reference)

## 2026-10-24

Timespent: 2-4 hours

Built the batchnorm naively and in the same API format of PyTorch. Understood and implemented diff diagnostic tools to see the activation outputs, each layer's grad, param's grad distribution, grad/param update ratio.. Using AI, I have got list of diff scenarious to test the model with and note the observations accordingly. I have to do that tomorrow.

## 2026-10-25

Timespent: 2 hours

Finished with the experiments and noted the same in the [Inferences decision tree.png](../Notes/Month%204/Inferences%20decision%20tree.png) image, [3 deep nn runs.md](../Month%204/3%20deep%20nn%20runs.md) file and in the [Notes4.md](../Notes/Notes%204.md) file as well.

Mark it as end of session 3 and 4 of month 4. For session 5-7, watch backprop ninja video and do only the backprop and see the hand derivations carefully (no need to implement everything).

- Tomorrow start with watching the video and start the backprop for some variables. I can't finish it fully
- Finish setting up the basic code to test the manual backprop calculations.

## 2026-10-26

Timespent: 3 hours

Finished with the manual backprop without watching the video (as I have already watched this months before). Though I took time, I was satisfied with my understanding. 

Tomorrow, make it a full runnable one and train the model and observe the losses. If possible, watch the remaining sections of the video, only after manual backprop implementation, get losses and print predictions.

Remaining section of the video: https://youtu.be/q8SA3rM6ckI?t=5190
Reference ipynb of Andrej: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb

## 2026-10-27

Timespent: 30 mins

Woke up late. Finished the coding part. Video yet to be seen once. Will watch it tomorrow and try to wrap up till the session 7, take notes and get started with session 8's wavenet understanding

## 2026-10-28

Timespent: 1hour 15 mins

Watched the video. Have a mental idea about how to implement the [wavenet](https://youtu.be/t3YJ5hKiMQ0). Do that tomorrow. Containerize everything, move everything to proper classes, implement the wavenet, take baseline readings to compare everything. If possible, try to answer the questions inthe [Month 4 curriculum](../Curriculum/Monthly%20Curriculum/Month4.md) file.

## 2026-10-29

Timespent: 30 mins

Woke up late. Implemented the wavenet model, but the loss is converging fast to near 0, but dev/test loss are bad. There was a problem in creating dataset because of the more context length 8, which caused problems in name like "emma", so I had to pad 8 "."s at the beginning of each name. Documented the loss and predictions of plain model and wavenet model.

Tomorrow, answer the questions in the curriculum and finish session 8 and 9.

## 2026-10-30

Timespent: 1 hour

Finished session 8,9. Was confused what is session 10. So created detailed curriculum for the same. Then started session 10 and finished upto part 4 (out of 10 parts).

Tomorrow, try to finish up more parts as I don't know how much time I'll take to get it done.

## 2026-10-31

Timespent: scattered 3 hours

Finished session 11 and written notes (with AI). 