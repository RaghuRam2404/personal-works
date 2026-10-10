
# Month 1 

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

# Month 2

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