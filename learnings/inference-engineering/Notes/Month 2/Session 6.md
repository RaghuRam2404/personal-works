Answer without notes:

What does autocast decide?
>> Which operation to run in float16/bfloat16 and which to run in float32

What does autocast not decide?
>> It does not decide when to scale gradients or handle underflow/overflow issues.

Why should backward happen outside the autocast context?
>> Because the forward pass may use AMP (automatic mixed precision), the logit we get or the loss we get may be in the lower precision. We have to scale the gradients during the backward pass to prevent underflow, if the fp16 is used.

Why does FP16 commonly need gradient scaling?
>> Because of underflow and limited precision in FP16, small gradient values may become zero, which can hinder learning.

Why does BF16 normally need less protection against underflow?
>> becase we lose precisions between 0 and 1 but not in the exponent part of the gradients, underflow won't happen.

Why does gradient scaling not intentionally change the final parameter update?
>> since it's a constant scaling factor, it doesn't alter the gradient of the parameters after unscaling.