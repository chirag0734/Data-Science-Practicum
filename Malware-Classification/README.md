# n00bs-project2

In this project, we have been given the task of Malware classification using bytes and their generated ASM files. We are free to use either or both. 

# Feature Generation

For Feature generation, we extracted two types of features out of the ASM files (ignoring the bytes files altogether). The two features extracted were: Opcodes AND Segment type. The feature vector consisted of: 
FEATURE_VECTOR = OPCODES [1-199] SEGMENTS[200-600]

With this transformation in place, we just tried classification with random forest and got a CV of ~97%. Upon trying it on the actual testing set, we got an accuracy of 99%+. 

# Code

The code is fairly simple and WILL RUN JUST FINE on a single machine (albeit slow). Successfully ran the code on the larger dataset in < 6 hours on my local machine with the data being accessed from S3. Code can be further optimized easily since right now it makes two passes through the asm file (once for bytecodes, second for segments). Makine a single pass should boost the speed easily. 
