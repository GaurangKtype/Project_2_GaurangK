е╜
Ъ¤
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

√
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02unknown8Ж├
А
Adam/dense_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_50/bias/v
y
(Adam/dense_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_50/bias/v*
_output_shapes
:*
dtype0
Й
Adam/dense_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_50/kernel/v
В
*Adam/dense_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_50/kernel/v*
_output_shapes
:	А*
dtype0
Б
Adam/dense_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_49/bias/v
z
(Adam/dense_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_49/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_49/kernel/v
Г
*Adam/dense_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_49/kernel/v* 
_output_shapes
:
АА*
dtype0
Ь
"Adam/batch_normalization_68/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_68/beta/v
Х
6Adam/batch_normalization_68/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_68/beta/v*
_output_shapes
:@*
dtype0
Ю
#Adam/batch_normalization_68/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_68/gamma/v
Ч
7Adam/batch_normalization_68/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_68/gamma/v*
_output_shapes
:@*
dtype0
В
Adam/conv2d_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_68/bias/v
{
)Adam/conv2d_68/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/bias/v*
_output_shapes
:@*
dtype0
У
Adam/conv2d_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*(
shared_nameAdam/conv2d_68/kernel/v
М
+Adam/conv2d_68/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/kernel/v*'
_output_shapes
:А@*
dtype0
Э
"Adam/batch_normalization_67/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_67/beta/v
Ц
6Adam/batch_normalization_67/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_67/beta/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_67/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_67/gamma/v
Ш
7Adam/batch_normalization_67/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_67/gamma/v*
_output_shapes	
:А*
dtype0
Г
Adam/conv2d_67/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_67/bias/v
|
)Adam/conv2d_67/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/bias/v*
_output_shapes	
:А*
dtype0
У
Adam/conv2d_67/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/conv2d_67/kernel/v
М
+Adam/conv2d_67/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/kernel/v*'
_output_shapes
:@А*
dtype0
Ь
"Adam/batch_normalization_66/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_66/beta/v
Х
6Adam/batch_normalization_66/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_66/beta/v*
_output_shapes
:@*
dtype0
Ю
#Adam/batch_normalization_66/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_66/gamma/v
Ч
7Adam/batch_normalization_66/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_66/gamma/v*
_output_shapes
:@*
dtype0
В
Adam/conv2d_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_66/bias/v
{
)Adam/conv2d_66/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/bias/v*
_output_shapes
:@*
dtype0
Т
Adam/conv2d_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_66/kernel/v
Л
+Adam/conv2d_66/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/kernel/v*&
_output_shapes
:@*
dtype0
А
Adam/dense_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_50/bias/m
y
(Adam/dense_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_50/bias/m*
_output_shapes
:*
dtype0
Й
Adam/dense_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_50/kernel/m
В
*Adam/dense_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_50/kernel/m*
_output_shapes
:	А*
dtype0
Б
Adam/dense_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_49/bias/m
z
(Adam/dense_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_49/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/dense_49/kernel/m
Г
*Adam/dense_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_49/kernel/m* 
_output_shapes
:
АА*
dtype0
Ь
"Adam/batch_normalization_68/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_68/beta/m
Х
6Adam/batch_normalization_68/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_68/beta/m*
_output_shapes
:@*
dtype0
Ю
#Adam/batch_normalization_68/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_68/gamma/m
Ч
7Adam/batch_normalization_68/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_68/gamma/m*
_output_shapes
:@*
dtype0
В
Adam/conv2d_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_68/bias/m
{
)Adam/conv2d_68/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/bias/m*
_output_shapes
:@*
dtype0
У
Adam/conv2d_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*(
shared_nameAdam/conv2d_68/kernel/m
М
+Adam/conv2d_68/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_68/kernel/m*'
_output_shapes
:А@*
dtype0
Э
"Adam/batch_normalization_67/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_67/beta/m
Ц
6Adam/batch_normalization_67/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_67/beta/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_67/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_67/gamma/m
Ш
7Adam/batch_normalization_67/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_67/gamma/m*
_output_shapes	
:А*
dtype0
Г
Adam/conv2d_67/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_67/bias/m
|
)Adam/conv2d_67/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/bias/m*
_output_shapes	
:А*
dtype0
У
Adam/conv2d_67/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*(
shared_nameAdam/conv2d_67/kernel/m
М
+Adam/conv2d_67/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_67/kernel/m*'
_output_shapes
:@А*
dtype0
Ь
"Adam/batch_normalization_66/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_66/beta/m
Х
6Adam/batch_normalization_66/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_66/beta/m*
_output_shapes
:@*
dtype0
Ю
#Adam/batch_normalization_66/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_66/gamma/m
Ч
7Adam/batch_normalization_66/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_66/gamma/m*
_output_shapes
:@*
dtype0
В
Adam/conv2d_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_66/bias/m
{
)Adam/conv2d_66/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/bias/m*
_output_shapes
:@*
dtype0
Т
Adam/conv2d_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_66/kernel/m
Л
+Adam/conv2d_66/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_66/kernel/m*&
_output_shapes
:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_50/bias
k
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes
:*
dtype0
{
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_50/kernel
t
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel*
_output_shapes
:	А*
dtype0
s
dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_49/bias
l
!dense_49/bias/Read/ReadVariableOpReadVariableOpdense_49/bias*
_output_shapes	
:А*
dtype0
|
dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_49/kernel
u
#dense_49/kernel/Read/ReadVariableOpReadVariableOpdense_49/kernel* 
_output_shapes
:
АА*
dtype0
д
&batch_normalization_68/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_68/moving_variance
Э
:batch_normalization_68/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_68/moving_variance*
_output_shapes
:@*
dtype0
Ь
"batch_normalization_68/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_68/moving_mean
Х
6batch_normalization_68/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_68/moving_mean*
_output_shapes
:@*
dtype0
О
batch_normalization_68/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_68/beta
З
/batch_normalization_68/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_68/beta*
_output_shapes
:@*
dtype0
Р
batch_normalization_68/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_68/gamma
Й
0batch_normalization_68/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_68/gamma*
_output_shapes
:@*
dtype0
t
conv2d_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_68/bias
m
"conv2d_68/bias/Read/ReadVariableOpReadVariableOpconv2d_68/bias*
_output_shapes
:@*
dtype0
Е
conv2d_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*!
shared_nameconv2d_68/kernel
~
$conv2d_68/kernel/Read/ReadVariableOpReadVariableOpconv2d_68/kernel*'
_output_shapes
:А@*
dtype0
е
&batch_normalization_67/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_67/moving_variance
Ю
:batch_normalization_67/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_67/moving_variance*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_67/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_67/moving_mean
Ц
6batch_normalization_67/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_67/moving_mean*
_output_shapes	
:А*
dtype0
П
batch_normalization_67/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_67/beta
И
/batch_normalization_67/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_67/beta*
_output_shapes	
:А*
dtype0
С
batch_normalization_67/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_67/gamma
К
0batch_normalization_67/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_67/gamma*
_output_shapes	
:А*
dtype0
u
conv2d_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_67/bias
n
"conv2d_67/bias/Read/ReadVariableOpReadVariableOpconv2d_67/bias*
_output_shapes	
:А*
dtype0
Е
conv2d_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*!
shared_nameconv2d_67/kernel
~
$conv2d_67/kernel/Read/ReadVariableOpReadVariableOpconv2d_67/kernel*'
_output_shapes
:@А*
dtype0
д
&batch_normalization_66/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_66/moving_variance
Э
:batch_normalization_66/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_66/moving_variance*
_output_shapes
:@*
dtype0
Ь
"batch_normalization_66/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_66/moving_mean
Х
6batch_normalization_66/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_66/moving_mean*
_output_shapes
:@*
dtype0
О
batch_normalization_66/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_66/beta
З
/batch_normalization_66/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_66/beta*
_output_shapes
:@*
dtype0
Р
batch_normalization_66/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_66/gamma
Й
0batch_normalization_66/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_66/gamma*
_output_shapes
:@*
dtype0
t
conv2d_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_66/bias
m
"conv2d_66/bias/Read/ReadVariableOpReadVariableOpconv2d_66/bias*
_output_shapes
:@*
dtype0
Д
conv2d_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_66/kernel
}
$conv2d_66/kernel/Read/ReadVariableOpReadVariableOpconv2d_66/kernel*&
_output_shapes
:@*
dtype0
Т
serving_default_conv2d_66_inputPlaceholder*/
_output_shapes
:         dd*
dtype0*$
shape:         dd
б
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_66_inputconv2d_66/kernelconv2d_66/biasbatch_normalization_66/gammabatch_normalization_66/beta"batch_normalization_66/moving_mean&batch_normalization_66/moving_varianceconv2d_67/kernelconv2d_67/biasbatch_normalization_67/gammabatch_normalization_67/beta"batch_normalization_67/moving_mean&batch_normalization_67/moving_varianceconv2d_68/kernelconv2d_68/biasbatch_normalization_68/gammabatch_normalization_68/beta"batch_normalization_68/moving_mean&batch_normalization_68/moving_variancedense_49/kerneldense_49/biasdense_50/kerneldense_50/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_270262

NoOpNoOp
╛З
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*°Ж
valueэЖBщЖ BсЖ
╚
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
╒
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&axis
	'gamma
(beta
)moving_mean
*moving_variance*
О
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
╚
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op*
╒
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance*
О
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
╚
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias
 S_jit_compiled_convolution_op*
╒
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance*
О
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
О
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses* 
ж
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias*
е
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
y_random_generator* 
и
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
Аkernel
	Бbias*
м
0
1
'2
(3
)4
*5
76
87
A8
B9
C10
D11
Q12
R13
[14
\15
]16
^17
q18
r19
А20
Б21*
|
0
1
'2
(3
74
85
A6
B7
Q8
R9
[10
\11
q12
r13
А14
Б15*


В0* 
╡
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Иtrace_0
Йtrace_1
Кtrace_2
Лtrace_3* 
:
Мtrace_0
Нtrace_1
Оtrace_2
Пtrace_3* 
* 
Н
	Рiter
Сbeta_1
Тbeta_2

Уdecay
Фlearning_ratemЕmЖ'mЗ(mИ7mЙ8mКAmЛBmМQmНRmО[mП\mРqmСrmТ	АmУ	БmФvХvЦ'vЧ(vШ7vЩ8vЪAvЫBvЬQvЭRvЮ[vЯ\vаqvбrvв	Аvг	Бvд*

Хserving_default* 

0
1*

0
1*
* 
Ш
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ыtrace_0* 

Ьtrace_0* 
`Z
VARIABLE_VALUEconv2d_66/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_66/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
'0
(1
)2
*3*

'0
(1*
* 
Ш
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

вtrace_0
гtrace_1* 

дtrace_0
еtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_66/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_66/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_66/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_66/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

лtrace_0* 

мtrace_0* 

70
81*

70
81*
* 
Ш
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

▓trace_0* 

│trace_0* 
`Z
VARIABLE_VALUEconv2d_67/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_67/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
A0
B1
C2
D3*

A0
B1*
* 
Ш
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

╣trace_0
║trace_1* 

╗trace_0
╝trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_67/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_67/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_67/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_67/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 

┬trace_0* 

├trace_0* 

Q0
R1*

Q0
R1*
* 
Ш
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

╔trace_0* 

╩trace_0* 
`Z
VARIABLE_VALUEconv2d_68/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_68/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
[0
\1
]2
^3*

[0
\1*
* 
Ш
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

╨trace_0
╤trace_1* 

╥trace_0
╙trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_68/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_68/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_68/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_68/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

┘trace_0* 

┌trace_0* 
* 
* 
* 
Ц
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

рtrace_0* 

сtrace_0* 

q0
r1*

q0
r1*


В0* 
Ш
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

чtrace_0* 

шtrace_0* 
_Y
VARIABLE_VALUEdense_49/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_49/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 

юtrace_0
яtrace_1* 

Ёtrace_0
ёtrace_1* 
* 

А0
Б1*

А0
Б1*
* 
Ш
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ўtrace_0* 

°trace_0* 
_Y
VARIABLE_VALUEdense_50/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_50/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

∙trace_0* 
.
)0
*1
C2
D3
]4
^5*
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*

·0
√1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

)0
*1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

C0
D1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

]0
^1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


В0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
№	variables
¤	keras_api

■total

 count*
M
А	variables
Б	keras_api

Вtotal

Гcount
Д
_fn_kwargs*

■0
 1*

№	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

В0
Г1*

А	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Г}
VARIABLE_VALUEAdam/conv2d_66/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_66/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_66/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_66/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/conv2d_67/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_67/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_67/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_67/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/conv2d_68/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_68/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_68/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_68/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_49/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_49/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_50/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_50/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/conv2d_66/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_66/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_66/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_66/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/conv2d_67/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_67/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_67/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_67/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/conv2d_68/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_68/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_68/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_68/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_49/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_49/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_50/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_50/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ї
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_66/kernel/Read/ReadVariableOp"conv2d_66/bias/Read/ReadVariableOp0batch_normalization_66/gamma/Read/ReadVariableOp/batch_normalization_66/beta/Read/ReadVariableOp6batch_normalization_66/moving_mean/Read/ReadVariableOp:batch_normalization_66/moving_variance/Read/ReadVariableOp$conv2d_67/kernel/Read/ReadVariableOp"conv2d_67/bias/Read/ReadVariableOp0batch_normalization_67/gamma/Read/ReadVariableOp/batch_normalization_67/beta/Read/ReadVariableOp6batch_normalization_67/moving_mean/Read/ReadVariableOp:batch_normalization_67/moving_variance/Read/ReadVariableOp$conv2d_68/kernel/Read/ReadVariableOp"conv2d_68/bias/Read/ReadVariableOp0batch_normalization_68/gamma/Read/ReadVariableOp/batch_normalization_68/beta/Read/ReadVariableOp6batch_normalization_68/moving_mean/Read/ReadVariableOp:batch_normalization_68/moving_variance/Read/ReadVariableOp#dense_49/kernel/Read/ReadVariableOp!dense_49/bias/Read/ReadVariableOp#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_66/kernel/m/Read/ReadVariableOp)Adam/conv2d_66/bias/m/Read/ReadVariableOp7Adam/batch_normalization_66/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_66/beta/m/Read/ReadVariableOp+Adam/conv2d_67/kernel/m/Read/ReadVariableOp)Adam/conv2d_67/bias/m/Read/ReadVariableOp7Adam/batch_normalization_67/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_67/beta/m/Read/ReadVariableOp+Adam/conv2d_68/kernel/m/Read/ReadVariableOp)Adam/conv2d_68/bias/m/Read/ReadVariableOp7Adam/batch_normalization_68/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_68/beta/m/Read/ReadVariableOp*Adam/dense_49/kernel/m/Read/ReadVariableOp(Adam/dense_49/bias/m/Read/ReadVariableOp*Adam/dense_50/kernel/m/Read/ReadVariableOp(Adam/dense_50/bias/m/Read/ReadVariableOp+Adam/conv2d_66/kernel/v/Read/ReadVariableOp)Adam/conv2d_66/bias/v/Read/ReadVariableOp7Adam/batch_normalization_66/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_66/beta/v/Read/ReadVariableOp+Adam/conv2d_67/kernel/v/Read/ReadVariableOp)Adam/conv2d_67/bias/v/Read/ReadVariableOp7Adam/batch_normalization_67/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_67/beta/v/Read/ReadVariableOp+Adam/conv2d_68/kernel/v/Read/ReadVariableOp)Adam/conv2d_68/bias/v/Read/ReadVariableOp7Adam/batch_normalization_68/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_68/beta/v/Read/ReadVariableOp*Adam/dense_49/kernel/v/Read/ReadVariableOp(Adam/dense_49/bias/v/Read/ReadVariableOp*Adam/dense_50/kernel/v/Read/ReadVariableOp(Adam/dense_50/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_271132
Г
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_66/kernelconv2d_66/biasbatch_normalization_66/gammabatch_normalization_66/beta"batch_normalization_66/moving_mean&batch_normalization_66/moving_varianceconv2d_67/kernelconv2d_67/biasbatch_normalization_67/gammabatch_normalization_67/beta"batch_normalization_67/moving_mean&batch_normalization_67/moving_varianceconv2d_68/kernelconv2d_68/biasbatch_normalization_68/gammabatch_normalization_68/beta"batch_normalization_68/moving_mean&batch_normalization_68/moving_variancedense_49/kerneldense_49/biasdense_50/kerneldense_50/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_66/kernel/mAdam/conv2d_66/bias/m#Adam/batch_normalization_66/gamma/m"Adam/batch_normalization_66/beta/mAdam/conv2d_67/kernel/mAdam/conv2d_67/bias/m#Adam/batch_normalization_67/gamma/m"Adam/batch_normalization_67/beta/mAdam/conv2d_68/kernel/mAdam/conv2d_68/bias/m#Adam/batch_normalization_68/gamma/m"Adam/batch_normalization_68/beta/mAdam/dense_49/kernel/mAdam/dense_49/bias/mAdam/dense_50/kernel/mAdam/dense_50/bias/mAdam/conv2d_66/kernel/vAdam/conv2d_66/bias/v#Adam/batch_normalization_66/gamma/v"Adam/batch_normalization_66/beta/vAdam/conv2d_67/kernel/vAdam/conv2d_67/bias/v#Adam/batch_normalization_67/gamma/v"Adam/batch_normalization_67/beta/vAdam/conv2d_68/kernel/vAdam/conv2d_68/bias/v#Adam/batch_normalization_68/gamma/v"Adam/batch_normalization_68/beta/vAdam/dense_49/kernel/vAdam/dense_49/bias/vAdam/dense_50/kernel/vAdam/dense_50/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_271331ЖЕ
▌
d
F__inference_dropout_27_layer_call_and_return_conditional_losses_269709

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ф
h
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_270829

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┐
═
.__inference_sequential_22_layer_call_fn_270364

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А%

unknown_11:А@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
АА

unknown_18:	А

unknown_19:	А

unknown_20:
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_269975o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
─p
Е
I__inference_sequential_22_layer_call_and_return_conditional_losses_270455

inputsB
(conv2d_66_conv2d_readvariableop_resource:@7
)conv2d_66_biasadd_readvariableop_resource:@<
.batch_normalization_66_readvariableop_resource:@>
0batch_normalization_66_readvariableop_1_resource:@M
?batch_normalization_66_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_67_conv2d_readvariableop_resource:@А8
)conv2d_67_biasadd_readvariableop_resource:	А=
.batch_normalization_67_readvariableop_resource:	А?
0batch_normalization_67_readvariableop_1_resource:	АN
?batch_normalization_67_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource:	АC
(conv2d_68_conv2d_readvariableop_resource:А@7
)conv2d_68_biasadd_readvariableop_resource:@<
.batch_normalization_68_readvariableop_resource:@>
0batch_normalization_68_readvariableop_1_resource:@M
?batch_normalization_68_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource:@;
'dense_49_matmul_readvariableop_resource:
АА7
(dense_49_biasadd_readvariableop_resource:	А:
'dense_50_matmul_readvariableop_resource:	А6
(dense_50_biasadd_readvariableop_resource:
identityИв6batch_normalization_66/FusedBatchNormV3/ReadVariableOpв8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_66/ReadVariableOpв'batch_normalization_66/ReadVariableOp_1в6batch_normalization_67/FusedBatchNormV3/ReadVariableOpв8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_67/ReadVariableOpв'batch_normalization_67/ReadVariableOp_1в6batch_normalization_68/FusedBatchNormV3/ReadVariableOpв8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_68/ReadVariableOpв'batch_normalization_68/ReadVariableOp_1в conv2d_66/BiasAdd/ReadVariableOpвconv2d_66/Conv2D/ReadVariableOpв conv2d_67/BiasAdd/ReadVariableOpвconv2d_67/Conv2D/ReadVariableOpв conv2d_68/BiasAdd/ReadVariableOpвconv2d_68/Conv2D/ReadVariableOpвdense_49/BiasAdd/ReadVariableOpвdense_49/MatMul/ReadVariableOpв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpвdense_50/BiasAdd/ReadVariableOpвdense_50/MatMul/ReadVariableOpР
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0н
conv2d_66/Conv2DConv2Dinputs'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_66/ReluReluconv2d_66/BiasAdd:output:0*
T0*/
_output_shapes
:         @Р
%batch_normalization_66/ReadVariableOpReadVariableOp.batch_normalization_66_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_66/ReadVariableOp_1ReadVariableOp0batch_normalization_66_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┐
'batch_normalization_66/FusedBatchNormV3FusedBatchNormV3conv2d_66/Relu:activations:0-batch_normalization_66/ReadVariableOp:value:0/batch_normalization_66/ReadVariableOp_1:value:0>batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( ╜
max_pooling2d_66/MaxPoolMaxPool+batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
С
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0╔
conv2d_67/Conv2DConv2D!max_pooling2d_66/MaxPool:output:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
З
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аm
conv2d_67/ReluReluconv2d_67/BiasAdd:output:0*
T0*0
_output_shapes
:         АС
%batch_normalization_67/ReadVariableOpReadVariableOp.batch_normalization_67_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_67/ReadVariableOp_1ReadVariableOp0batch_normalization_67_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_67/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_67_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0─
'batch_normalization_67/FusedBatchNormV3FusedBatchNormV3conv2d_67/Relu:activations:0-batch_normalization_67/ReadVariableOp:value:0/batch_normalization_67/ReadVariableOp_1:value:0>batch_normalization_67/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( ╛
max_pooling2d_67/MaxPoolMaxPool+batch_normalization_67/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
С
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0╚
conv2d_68/Conv2DConv2D!max_pooling2d_67/MaxPool:output:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_68/ReluReluconv2d_68/BiasAdd:output:0*
T0*/
_output_shapes
:         @Р
%batch_normalization_68/ReadVariableOpReadVariableOp.batch_normalization_68_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_68/ReadVariableOp_1ReadVariableOp0batch_normalization_68_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_68/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_68_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┐
'batch_normalization_68/FusedBatchNormV3FusedBatchNormV3conv2d_68/Relu:activations:0-batch_normalization_68/ReadVariableOp:value:0/batch_normalization_68/ReadVariableOp_1:value:0>batch_normalization_68/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( ╜
max_pooling2d_68/MaxPoolMaxPool+batch_normalization_68/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
a
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"       О
flatten_22/ReshapeReshape!max_pooling2d_68/MaxPool:output:0flatten_22/Const:output:0*
T0*(
_output_shapes
:         АИ
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0С
dense_49/MatMulMatMulflatten_22/Reshape:output:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*(
_output_shapes
:         Аo
dropout_27/IdentityIdentitydense_49/Relu:activations:0*
T0*(
_output_shapes
:         АЗ
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0С
dense_50/MatMulMatMuldropout_27/Identity:output:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_50/SoftmaxSoftmaxdense_50/BiasAdd:output:0*
T0*'
_output_shapes
:         Ы
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_50/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         б
NoOpNoOp7^batch_normalization_66/FusedBatchNormV3/ReadVariableOp9^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_66/ReadVariableOp(^batch_normalization_66/ReadVariableOp_17^batch_normalization_67/FusedBatchNormV3/ReadVariableOp9^batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_67/ReadVariableOp(^batch_normalization_67/ReadVariableOp_17^batch_normalization_68/FusedBatchNormV3/ReadVariableOp9^batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_68/ReadVariableOp(^batch_normalization_68/ReadVariableOp_1!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_66/FusedBatchNormV3/ReadVariableOp6batch_normalization_66/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_18batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_66/ReadVariableOp%batch_normalization_66/ReadVariableOp2R
'batch_normalization_66/ReadVariableOp_1'batch_normalization_66/ReadVariableOp_12p
6batch_normalization_67/FusedBatchNormV3/ReadVariableOp6batch_normalization_67/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_18batch_normalization_67/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_67/ReadVariableOp%batch_normalization_67/ReadVariableOp2R
'batch_normalization_67/ReadVariableOp_1'batch_normalization_67/ReadVariableOp_12p
6batch_normalization_68/FusedBatchNormV3/ReadVariableOp6batch_normalization_68/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_18batch_normalization_68/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_68/ReadVariableOp%batch_normalization_68/ReadVariableOp2R
'batch_normalization_68/ReadVariableOp_1'batch_normalization_68/ReadVariableOp_12D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
У	
╥
7__inference_batch_normalization_68_layer_call_fn_270770

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_269533Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
я
а
*__inference_conv2d_68_layer_call_fn_270746

inputs"
unknown:А@
	unknown_0:@
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_269659w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_270819

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
е
G
+__inference_dropout_27_layer_call_fn_270869

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_269709a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ШО
√
I__inference_sequential_22_layer_call_and_return_conditional_losses_270553

inputsB
(conv2d_66_conv2d_readvariableop_resource:@7
)conv2d_66_biasadd_readvariableop_resource:@<
.batch_normalization_66_readvariableop_resource:@>
0batch_normalization_66_readvariableop_1_resource:@M
?batch_normalization_66_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_67_conv2d_readvariableop_resource:@А8
)conv2d_67_biasadd_readvariableop_resource:	А=
.batch_normalization_67_readvariableop_resource:	А?
0batch_normalization_67_readvariableop_1_resource:	АN
?batch_normalization_67_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource:	АC
(conv2d_68_conv2d_readvariableop_resource:А@7
)conv2d_68_biasadd_readvariableop_resource:@<
.batch_normalization_68_readvariableop_resource:@>
0batch_normalization_68_readvariableop_1_resource:@M
?batch_normalization_68_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource:@;
'dense_49_matmul_readvariableop_resource:
АА7
(dense_49_biasadd_readvariableop_resource:	А:
'dense_50_matmul_readvariableop_resource:	А6
(dense_50_biasadd_readvariableop_resource:
identityИв%batch_normalization_66/AssignNewValueв'batch_normalization_66/AssignNewValue_1в6batch_normalization_66/FusedBatchNormV3/ReadVariableOpв8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_66/ReadVariableOpв'batch_normalization_66/ReadVariableOp_1в%batch_normalization_67/AssignNewValueв'batch_normalization_67/AssignNewValue_1в6batch_normalization_67/FusedBatchNormV3/ReadVariableOpв8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_67/ReadVariableOpв'batch_normalization_67/ReadVariableOp_1в%batch_normalization_68/AssignNewValueв'batch_normalization_68/AssignNewValue_1в6batch_normalization_68/FusedBatchNormV3/ReadVariableOpв8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_68/ReadVariableOpв'batch_normalization_68/ReadVariableOp_1в conv2d_66/BiasAdd/ReadVariableOpвconv2d_66/Conv2D/ReadVariableOpв conv2d_67/BiasAdd/ReadVariableOpвconv2d_67/Conv2D/ReadVariableOpв conv2d_68/BiasAdd/ReadVariableOpвconv2d_68/Conv2D/ReadVariableOpвdense_49/BiasAdd/ReadVariableOpвdense_49/MatMul/ReadVariableOpв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpвdense_50/BiasAdd/ReadVariableOpвdense_50/MatMul/ReadVariableOpР
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0н
conv2d_66/Conv2DConv2Dinputs'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_66/ReluReluconv2d_66/BiasAdd:output:0*
T0*/
_output_shapes
:         @Р
%batch_normalization_66/ReadVariableOpReadVariableOp.batch_normalization_66_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_66/ReadVariableOp_1ReadVariableOp0batch_normalization_66_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0═
'batch_normalization_66/FusedBatchNormV3FusedBatchNormV3conv2d_66/Relu:activations:0-batch_normalization_66/ReadVariableOp:value:0/batch_normalization_66/ReadVariableOp_1:value:0>batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_66/AssignNewValueAssignVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource4batch_normalization_66/FusedBatchNormV3:batch_mean:07^batch_normalization_66/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_66/AssignNewValue_1AssignVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_66/FusedBatchNormV3:batch_variance:09^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(╜
max_pooling2d_66/MaxPoolMaxPool+batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
С
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0╔
conv2d_67/Conv2DConv2D!max_pooling2d_66/MaxPool:output:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
З
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аm
conv2d_67/ReluReluconv2d_67/BiasAdd:output:0*
T0*0
_output_shapes
:         АС
%batch_normalization_67/ReadVariableOpReadVariableOp.batch_normalization_67_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_67/ReadVariableOp_1ReadVariableOp0batch_normalization_67_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_67/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_67_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╥
'batch_normalization_67/FusedBatchNormV3FusedBatchNormV3conv2d_67/Relu:activations:0-batch_normalization_67/ReadVariableOp:value:0/batch_normalization_67/ReadVariableOp_1:value:0>batch_normalization_67/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_67/AssignNewValueAssignVariableOp?batch_normalization_67_fusedbatchnormv3_readvariableop_resource4batch_normalization_67/FusedBatchNormV3:batch_mean:07^batch_normalization_67/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_67/AssignNewValue_1AssignVariableOpAbatch_normalization_67_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_67/FusedBatchNormV3:batch_variance:09^batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(╛
max_pooling2d_67/MaxPoolMaxPool+batch_normalization_67/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
С
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0╚
conv2d_68/Conv2DConv2D!max_pooling2d_67/MaxPool:output:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_68/ReluReluconv2d_68/BiasAdd:output:0*
T0*/
_output_shapes
:         @Р
%batch_normalization_68/ReadVariableOpReadVariableOp.batch_normalization_68_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_68/ReadVariableOp_1ReadVariableOp0batch_normalization_68_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_68/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_68_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0═
'batch_normalization_68/FusedBatchNormV3FusedBatchNormV3conv2d_68/Relu:activations:0-batch_normalization_68/ReadVariableOp:value:0/batch_normalization_68/ReadVariableOp_1:value:0>batch_normalization_68/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_68/AssignNewValueAssignVariableOp?batch_normalization_68_fusedbatchnormv3_readvariableop_resource4batch_normalization_68/FusedBatchNormV3:batch_mean:07^batch_normalization_68/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_68/AssignNewValue_1AssignVariableOpAbatch_normalization_68_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_68/FusedBatchNormV3:batch_variance:09^batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(╜
max_pooling2d_68/MaxPoolMaxPool+batch_normalization_68/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
a
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"       О
flatten_22/ReshapeReshape!max_pooling2d_68/MaxPool:output:0flatten_22/Const:output:0*
T0*(
_output_shapes
:         АИ
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0С
dense_49/MatMulMatMulflatten_22/Reshape:output:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*(
_output_shapes
:         А]
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Р
dropout_27/dropout/MulMuldense_49/Relu:activations:0!dropout_27/dropout/Const:output:0*
T0*(
_output_shapes
:         Аc
dropout_27/dropout/ShapeShapedense_49/Relu:activations:0*
T0*
_output_shapes
:г
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0f
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╚
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АЖ
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         АЛ
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*(
_output_shapes
:         АЗ
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0С
dense_50/MatMulMatMuldropout_27/dropout/Mul_1:z:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_50/SoftmaxSoftmaxdense_50/BiasAdd:output:0*
T0*'
_output_shapes
:         Ы
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_50/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         Ч

NoOpNoOp&^batch_normalization_66/AssignNewValue(^batch_normalization_66/AssignNewValue_17^batch_normalization_66/FusedBatchNormV3/ReadVariableOp9^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_66/ReadVariableOp(^batch_normalization_66/ReadVariableOp_1&^batch_normalization_67/AssignNewValue(^batch_normalization_67/AssignNewValue_17^batch_normalization_67/FusedBatchNormV3/ReadVariableOp9^batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_67/ReadVariableOp(^batch_normalization_67/ReadVariableOp_1&^batch_normalization_68/AssignNewValue(^batch_normalization_68/AssignNewValue_17^batch_normalization_68/FusedBatchNormV3/ReadVariableOp9^batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_68/ReadVariableOp(^batch_normalization_68/ReadVariableOp_1!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_66/AssignNewValue%batch_normalization_66/AssignNewValue2R
'batch_normalization_66/AssignNewValue_1'batch_normalization_66/AssignNewValue_12p
6batch_normalization_66/FusedBatchNormV3/ReadVariableOp6batch_normalization_66/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_18batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_66/ReadVariableOp%batch_normalization_66/ReadVariableOp2R
'batch_normalization_66/ReadVariableOp_1'batch_normalization_66/ReadVariableOp_12N
%batch_normalization_67/AssignNewValue%batch_normalization_67/AssignNewValue2R
'batch_normalization_67/AssignNewValue_1'batch_normalization_67/AssignNewValue_12p
6batch_normalization_67/FusedBatchNormV3/ReadVariableOp6batch_normalization_67/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_67/FusedBatchNormV3/ReadVariableOp_18batch_normalization_67/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_67/ReadVariableOp%batch_normalization_67/ReadVariableOp2R
'batch_normalization_67/ReadVariableOp_1'batch_normalization_67/ReadVariableOp_12N
%batch_normalization_68/AssignNewValue%batch_normalization_68/AssignNewValue2R
'batch_normalization_68/AssignNewValue_1'batch_normalization_68/AssignNewValue_12p
6batch_normalization_68/FusedBatchNormV3/ReadVariableOp6batch_normalization_68/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_68/FusedBatchNormV3/ReadVariableOp_18batch_normalization_68/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_68/ReadVariableOp%batch_normalization_68/ReadVariableOp2R
'batch_normalization_68/ReadVariableOp_1'batch_normalization_68/ReadVariableOp_12D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
╚
b
F__inference_flatten_22_layer_call_and_return_conditional_losses_269681

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╚
╗
__inference__traced_save_271132
file_prefix/
+savev2_conv2d_66_kernel_read_readvariableop-
)savev2_conv2d_66_bias_read_readvariableop;
7savev2_batch_normalization_66_gamma_read_readvariableop:
6savev2_batch_normalization_66_beta_read_readvariableopA
=savev2_batch_normalization_66_moving_mean_read_readvariableopE
Asavev2_batch_normalization_66_moving_variance_read_readvariableop/
+savev2_conv2d_67_kernel_read_readvariableop-
)savev2_conv2d_67_bias_read_readvariableop;
7savev2_batch_normalization_67_gamma_read_readvariableop:
6savev2_batch_normalization_67_beta_read_readvariableopA
=savev2_batch_normalization_67_moving_mean_read_readvariableopE
Asavev2_batch_normalization_67_moving_variance_read_readvariableop/
+savev2_conv2d_68_kernel_read_readvariableop-
)savev2_conv2d_68_bias_read_readvariableop;
7savev2_batch_normalization_68_gamma_read_readvariableop:
6savev2_batch_normalization_68_beta_read_readvariableopA
=savev2_batch_normalization_68_moving_mean_read_readvariableopE
Asavev2_batch_normalization_68_moving_variance_read_readvariableop.
*savev2_dense_49_kernel_read_readvariableop,
(savev2_dense_49_bias_read_readvariableop.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_66_kernel_m_read_readvariableop4
0savev2_adam_conv2d_66_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_66_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_66_beta_m_read_readvariableop6
2savev2_adam_conv2d_67_kernel_m_read_readvariableop4
0savev2_adam_conv2d_67_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_67_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_67_beta_m_read_readvariableop6
2savev2_adam_conv2d_68_kernel_m_read_readvariableop4
0savev2_adam_conv2d_68_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_68_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_68_beta_m_read_readvariableop5
1savev2_adam_dense_49_kernel_m_read_readvariableop3
/savev2_adam_dense_49_bias_m_read_readvariableop5
1savev2_adam_dense_50_kernel_m_read_readvariableop3
/savev2_adam_dense_50_bias_m_read_readvariableop6
2savev2_adam_conv2d_66_kernel_v_read_readvariableop4
0savev2_adam_conv2d_66_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_66_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_66_beta_v_read_readvariableop6
2savev2_adam_conv2d_67_kernel_v_read_readvariableop4
0savev2_adam_conv2d_67_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_67_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_67_beta_v_read_readvariableop6
2savev2_adam_conv2d_68_kernel_v_read_readvariableop4
0savev2_adam_conv2d_68_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_68_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_68_beta_v_read_readvariableop5
1savev2_adam_dense_49_kernel_v_read_readvariableop3
/savev2_adam_dense_49_bias_v_read_readvariableop5
1savev2_adam_dense_50_kernel_v_read_readvariableop3
/savev2_adam_dense_50_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: И#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*▒"
valueз"Bд"@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЁ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Х
valueЛBИ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ┬
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_66_kernel_read_readvariableop)savev2_conv2d_66_bias_read_readvariableop7savev2_batch_normalization_66_gamma_read_readvariableop6savev2_batch_normalization_66_beta_read_readvariableop=savev2_batch_normalization_66_moving_mean_read_readvariableopAsavev2_batch_normalization_66_moving_variance_read_readvariableop+savev2_conv2d_67_kernel_read_readvariableop)savev2_conv2d_67_bias_read_readvariableop7savev2_batch_normalization_67_gamma_read_readvariableop6savev2_batch_normalization_67_beta_read_readvariableop=savev2_batch_normalization_67_moving_mean_read_readvariableopAsavev2_batch_normalization_67_moving_variance_read_readvariableop+savev2_conv2d_68_kernel_read_readvariableop)savev2_conv2d_68_bias_read_readvariableop7savev2_batch_normalization_68_gamma_read_readvariableop6savev2_batch_normalization_68_beta_read_readvariableop=savev2_batch_normalization_68_moving_mean_read_readvariableopAsavev2_batch_normalization_68_moving_variance_read_readvariableop*savev2_dense_49_kernel_read_readvariableop(savev2_dense_49_bias_read_readvariableop*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_66_kernel_m_read_readvariableop0savev2_adam_conv2d_66_bias_m_read_readvariableop>savev2_adam_batch_normalization_66_gamma_m_read_readvariableop=savev2_adam_batch_normalization_66_beta_m_read_readvariableop2savev2_adam_conv2d_67_kernel_m_read_readvariableop0savev2_adam_conv2d_67_bias_m_read_readvariableop>savev2_adam_batch_normalization_67_gamma_m_read_readvariableop=savev2_adam_batch_normalization_67_beta_m_read_readvariableop2savev2_adam_conv2d_68_kernel_m_read_readvariableop0savev2_adam_conv2d_68_bias_m_read_readvariableop>savev2_adam_batch_normalization_68_gamma_m_read_readvariableop=savev2_adam_batch_normalization_68_beta_m_read_readvariableop1savev2_adam_dense_49_kernel_m_read_readvariableop/savev2_adam_dense_49_bias_m_read_readvariableop1savev2_adam_dense_50_kernel_m_read_readvariableop/savev2_adam_dense_50_bias_m_read_readvariableop2savev2_adam_conv2d_66_kernel_v_read_readvariableop0savev2_adam_conv2d_66_bias_v_read_readvariableop>savev2_adam_batch_normalization_66_gamma_v_read_readvariableop=savev2_adam_batch_normalization_66_beta_v_read_readvariableop2savev2_adam_conv2d_67_kernel_v_read_readvariableop0savev2_adam_conv2d_67_bias_v_read_readvariableop>savev2_adam_batch_normalization_67_gamma_v_read_readvariableop=savev2_adam_batch_normalization_67_beta_v_read_readvariableop2savev2_adam_conv2d_68_kernel_v_read_readvariableop0savev2_adam_conv2d_68_bias_v_read_readvariableop>savev2_adam_batch_normalization_68_gamma_v_read_readvariableop=savev2_adam_batch_normalization_68_beta_v_read_readvariableop1savev2_adam_dense_49_kernel_v_read_readvariableop/savev2_adam_dense_49_bias_v_read_readvariableop1savev2_adam_dense_50_kernel_v_read_readvariableop/savev2_adam_dense_50_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Р
_input_shapes■
√: :@:@:@:@:@:@:@А:А:А:А:А:А:А@:@:@:@:@:@:
АА:А:	А:: : : : : : : : : :@:@:@:@:@А:А:А:А:А@:@:@:@:
АА:А:	А::@:@:@:@:@А:А:А:А:А@:@:@:@:
АА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:!	

_output_shapes	
:А:!


_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:А@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@:-$)
'
_output_shapes
:@А:!%

_output_shapes	
:А:!&

_output_shapes	
:А:!'

_output_shapes	
:А:-()
'
_output_shapes
:А@: )

_output_shapes
:@: *

_output_shapes
:@: +

_output_shapes
:@:&,"
 
_output_shapes
:
АА:!-

_output_shapes	
:А:%.!

_output_shapes
:	А: /

_output_shapes
::,0(
&
_output_shapes
:@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:-4)
'
_output_shapes
:@А:!5

_output_shapes	
:А:!6

_output_shapes	
:А:!7

_output_shapes	
:А:-8)
'
_output_shapes
:А@: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:&<"
 
_output_shapes
:
АА:!=

_output_shapes	
:А:%>!

_output_shapes
:	А: ?

_output_shapes
::@

_output_shapes
: 
■В
Ъ
!__inference__wrapped_model_269359
conv2d_66_inputP
6sequential_22_conv2d_66_conv2d_readvariableop_resource:@E
7sequential_22_conv2d_66_biasadd_readvariableop_resource:@J
<sequential_22_batch_normalization_66_readvariableop_resource:@L
>sequential_22_batch_normalization_66_readvariableop_1_resource:@[
Msequential_22_batch_normalization_66_fusedbatchnormv3_readvariableop_resource:@]
Osequential_22_batch_normalization_66_fusedbatchnormv3_readvariableop_1_resource:@Q
6sequential_22_conv2d_67_conv2d_readvariableop_resource:@АF
7sequential_22_conv2d_67_biasadd_readvariableop_resource:	АK
<sequential_22_batch_normalization_67_readvariableop_resource:	АM
>sequential_22_batch_normalization_67_readvariableop_1_resource:	А\
Msequential_22_batch_normalization_67_fusedbatchnormv3_readvariableop_resource:	А^
Osequential_22_batch_normalization_67_fusedbatchnormv3_readvariableop_1_resource:	АQ
6sequential_22_conv2d_68_conv2d_readvariableop_resource:А@E
7sequential_22_conv2d_68_biasadd_readvariableop_resource:@J
<sequential_22_batch_normalization_68_readvariableop_resource:@L
>sequential_22_batch_normalization_68_readvariableop_1_resource:@[
Msequential_22_batch_normalization_68_fusedbatchnormv3_readvariableop_resource:@]
Osequential_22_batch_normalization_68_fusedbatchnormv3_readvariableop_1_resource:@I
5sequential_22_dense_49_matmul_readvariableop_resource:
ААE
6sequential_22_dense_49_biasadd_readvariableop_resource:	АH
5sequential_22_dense_50_matmul_readvariableop_resource:	АD
6sequential_22_dense_50_biasadd_readvariableop_resource:
identityИвDsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOpвFsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1в3sequential_22/batch_normalization_66/ReadVariableOpв5sequential_22/batch_normalization_66/ReadVariableOp_1вDsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOpвFsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1в3sequential_22/batch_normalization_67/ReadVariableOpв5sequential_22/batch_normalization_67/ReadVariableOp_1вDsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOpвFsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1в3sequential_22/batch_normalization_68/ReadVariableOpв5sequential_22/batch_normalization_68/ReadVariableOp_1в.sequential_22/conv2d_66/BiasAdd/ReadVariableOpв-sequential_22/conv2d_66/Conv2D/ReadVariableOpв.sequential_22/conv2d_67/BiasAdd/ReadVariableOpв-sequential_22/conv2d_67/Conv2D/ReadVariableOpв.sequential_22/conv2d_68/BiasAdd/ReadVariableOpв-sequential_22/conv2d_68/Conv2D/ReadVariableOpв-sequential_22/dense_49/BiasAdd/ReadVariableOpв,sequential_22/dense_49/MatMul/ReadVariableOpв-sequential_22/dense_50/BiasAdd/ReadVariableOpв,sequential_22/dense_50/MatMul/ReadVariableOpм
-sequential_22/conv2d_66/Conv2D/ReadVariableOpReadVariableOp6sequential_22_conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╥
sequential_22/conv2d_66/Conv2DConv2Dconv2d_66_input5sequential_22/conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
в
.sequential_22/conv2d_66/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv2d_66_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┼
sequential_22/conv2d_66/BiasAddBiasAdd'sequential_22/conv2d_66/Conv2D:output:06sequential_22/conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @И
sequential_22/conv2d_66/ReluRelu(sequential_22/conv2d_66/BiasAdd:output:0*
T0*/
_output_shapes
:         @м
3sequential_22/batch_normalization_66/ReadVariableOpReadVariableOp<sequential_22_batch_normalization_66_readvariableop_resource*
_output_shapes
:@*
dtype0░
5sequential_22/batch_normalization_66/ReadVariableOp_1ReadVariableOp>sequential_22_batch_normalization_66_readvariableop_1_resource*
_output_shapes
:@*
dtype0╬
Dsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_22_batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╥
Fsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_22_batch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0У
5sequential_22/batch_normalization_66/FusedBatchNormV3FusedBatchNormV3*sequential_22/conv2d_66/Relu:activations:0;sequential_22/batch_normalization_66/ReadVariableOp:value:0=sequential_22/batch_normalization_66/ReadVariableOp_1:value:0Lsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( ┘
&sequential_22/max_pooling2d_66/MaxPoolMaxPool9sequential_22/batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
н
-sequential_22/conv2d_67/Conv2D/ReadVariableOpReadVariableOp6sequential_22_conv2d_67_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0є
sequential_22/conv2d_67/Conv2DConv2D/sequential_22/max_pooling2d_66/MaxPool:output:05sequential_22/conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
г
.sequential_22/conv2d_67/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv2d_67_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╞
sequential_22/conv2d_67/BiasAddBiasAdd'sequential_22/conv2d_67/Conv2D:output:06sequential_22/conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЙ
sequential_22/conv2d_67/ReluRelu(sequential_22/conv2d_67/BiasAdd:output:0*
T0*0
_output_shapes
:         Ан
3sequential_22/batch_normalization_67/ReadVariableOpReadVariableOp<sequential_22_batch_normalization_67_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
5sequential_22/batch_normalization_67/ReadVariableOp_1ReadVariableOp>sequential_22_batch_normalization_67_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╧
Dsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_22_batch_normalization_67_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╙
Fsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_22_batch_normalization_67_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ш
5sequential_22/batch_normalization_67/FusedBatchNormV3FusedBatchNormV3*sequential_22/conv2d_67/Relu:activations:0;sequential_22/batch_normalization_67/ReadVariableOp:value:0=sequential_22/batch_normalization_67/ReadVariableOp_1:value:0Lsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( ┌
&sequential_22/max_pooling2d_67/MaxPoolMaxPool9sequential_22/batch_normalization_67/FusedBatchNormV3:y:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
н
-sequential_22/conv2d_68/Conv2D/ReadVariableOpReadVariableOp6sequential_22_conv2d_68_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Є
sequential_22/conv2d_68/Conv2DConv2D/sequential_22/max_pooling2d_67/MaxPool:output:05sequential_22/conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
в
.sequential_22/conv2d_68/BiasAdd/ReadVariableOpReadVariableOp7sequential_22_conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┼
sequential_22/conv2d_68/BiasAddBiasAdd'sequential_22/conv2d_68/Conv2D:output:06sequential_22/conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @И
sequential_22/conv2d_68/ReluRelu(sequential_22/conv2d_68/BiasAdd:output:0*
T0*/
_output_shapes
:         @м
3sequential_22/batch_normalization_68/ReadVariableOpReadVariableOp<sequential_22_batch_normalization_68_readvariableop_resource*
_output_shapes
:@*
dtype0░
5sequential_22/batch_normalization_68/ReadVariableOp_1ReadVariableOp>sequential_22_batch_normalization_68_readvariableop_1_resource*
_output_shapes
:@*
dtype0╬
Dsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_22_batch_normalization_68_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╥
Fsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_22_batch_normalization_68_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0У
5sequential_22/batch_normalization_68/FusedBatchNormV3FusedBatchNormV3*sequential_22/conv2d_68/Relu:activations:0;sequential_22/batch_normalization_68/ReadVariableOp:value:0=sequential_22/batch_normalization_68/ReadVariableOp_1:value:0Lsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( ┘
&sequential_22/max_pooling2d_68/MaxPoolMaxPool9sequential_22/batch_normalization_68/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
o
sequential_22/flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ╕
 sequential_22/flatten_22/ReshapeReshape/sequential_22/max_pooling2d_68/MaxPool:output:0'sequential_22/flatten_22/Const:output:0*
T0*(
_output_shapes
:         Ад
,sequential_22/dense_49/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_49_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╗
sequential_22/dense_49/MatMulMatMul)sequential_22/flatten_22/Reshape:output:04sequential_22/dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
-sequential_22/dense_49/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_49_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_22/dense_49/BiasAddBiasAdd'sequential_22/dense_49/MatMul:product:05sequential_22/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_22/dense_49/ReluRelu'sequential_22/dense_49/BiasAdd:output:0*
T0*(
_output_shapes
:         АЛ
!sequential_22/dropout_27/IdentityIdentity)sequential_22/dense_49/Relu:activations:0*
T0*(
_output_shapes
:         Аг
,sequential_22/dense_50/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_50_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0╗
sequential_22/dense_50/MatMulMatMul*sequential_22/dropout_27/Identity:output:04sequential_22/dense_50/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_22/dense_50/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_22/dense_50/BiasAddBiasAdd'sequential_22/dense_50/MatMul:product:05sequential_22/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
sequential_22/dense_50/SoftmaxSoftmax'sequential_22/dense_50/BiasAdd:output:0*
T0*'
_output_shapes
:         w
IdentityIdentity(sequential_22/dense_50/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         б

NoOpNoOpE^sequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOpG^sequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_14^sequential_22/batch_normalization_66/ReadVariableOp6^sequential_22/batch_normalization_66/ReadVariableOp_1E^sequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOpG^sequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_14^sequential_22/batch_normalization_67/ReadVariableOp6^sequential_22/batch_normalization_67/ReadVariableOp_1E^sequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOpG^sequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_14^sequential_22/batch_normalization_68/ReadVariableOp6^sequential_22/batch_normalization_68/ReadVariableOp_1/^sequential_22/conv2d_66/BiasAdd/ReadVariableOp.^sequential_22/conv2d_66/Conv2D/ReadVariableOp/^sequential_22/conv2d_67/BiasAdd/ReadVariableOp.^sequential_22/conv2d_67/Conv2D/ReadVariableOp/^sequential_22/conv2d_68/BiasAdd/ReadVariableOp.^sequential_22/conv2d_68/Conv2D/ReadVariableOp.^sequential_22/dense_49/BiasAdd/ReadVariableOp-^sequential_22/dense_49/MatMul/ReadVariableOp.^sequential_22/dense_50/BiasAdd/ReadVariableOp-^sequential_22/dense_50/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 2М
Dsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOpDsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1Fsequential_22/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12j
3sequential_22/batch_normalization_66/ReadVariableOp3sequential_22/batch_normalization_66/ReadVariableOp2n
5sequential_22/batch_normalization_66/ReadVariableOp_15sequential_22/batch_normalization_66/ReadVariableOp_12М
Dsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOpDsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_1Fsequential_22/batch_normalization_67/FusedBatchNormV3/ReadVariableOp_12j
3sequential_22/batch_normalization_67/ReadVariableOp3sequential_22/batch_normalization_67/ReadVariableOp2n
5sequential_22/batch_normalization_67/ReadVariableOp_15sequential_22/batch_normalization_67/ReadVariableOp_12М
Dsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOpDsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOp2Р
Fsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_1Fsequential_22/batch_normalization_68/FusedBatchNormV3/ReadVariableOp_12j
3sequential_22/batch_normalization_68/ReadVariableOp3sequential_22/batch_normalization_68/ReadVariableOp2n
5sequential_22/batch_normalization_68/ReadVariableOp_15sequential_22/batch_normalization_68/ReadVariableOp_12`
.sequential_22/conv2d_66/BiasAdd/ReadVariableOp.sequential_22/conv2d_66/BiasAdd/ReadVariableOp2^
-sequential_22/conv2d_66/Conv2D/ReadVariableOp-sequential_22/conv2d_66/Conv2D/ReadVariableOp2`
.sequential_22/conv2d_67/BiasAdd/ReadVariableOp.sequential_22/conv2d_67/BiasAdd/ReadVariableOp2^
-sequential_22/conv2d_67/Conv2D/ReadVariableOp-sequential_22/conv2d_67/Conv2D/ReadVariableOp2`
.sequential_22/conv2d_68/BiasAdd/ReadVariableOp.sequential_22/conv2d_68/BiasAdd/ReadVariableOp2^
-sequential_22/conv2d_68/Conv2D/ReadVariableOp-sequential_22/conv2d_68/Conv2D/ReadVariableOp2^
-sequential_22/dense_49/BiasAdd/ReadVariableOp-sequential_22/dense_49/BiasAdd/ReadVariableOp2\
,sequential_22/dense_49/MatMul/ReadVariableOp,sequential_22/dense_49/MatMul/ReadVariableOp2^
-sequential_22/dense_50/BiasAdd/ReadVariableOp-sequential_22/dense_50/BiasAdd/ReadVariableOp2\
,sequential_22/dense_50/MatMul/ReadVariableOp,sequential_22/dense_50/MatMul/ReadVariableOp:` \
/
_output_shapes
:         dd
)
_user_specified_nameconv2d_66_input
Ф
h
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_270737

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
М
А
E__inference_conv2d_67_layer_call_and_return_conditional_losses_269632

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ф
h
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_269584

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
У	
╥
7__inference_batch_normalization_66_layer_call_fn_270586

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_269381Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
И
 
E__inference_conv2d_68_layer_call_and_return_conditional_losses_269659

inputs9
conv2d_readvariableop_resource:А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
И
 
E__inference_conv2d_68_layer_call_and_return_conditional_losses_270757

inputs9
conv2d_readvariableop_resource:А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
│
G
+__inference_flatten_22_layer_call_fn_270834

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_22_layer_call_and_return_conditional_losses_269681a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╚
b
F__inference_flatten_22_layer_call_and_return_conditional_losses_270840

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
▌
б
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_270709

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ь
Я
*__inference_conv2d_66_layer_call_fn_270562

inputs!
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_269605w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
д	
│
__inference_loss_fn_0_270920N
:dense_49_kernel_regularizer_l2loss_readvariableop_resource:
АА
identityИв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpо
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_49_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_49/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp
д

Ў
D__inference_dense_50_layer_call_and_return_conditional_losses_269722

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ё
б
*__inference_conv2d_67_layer_call_fn_270654

inputs"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_269632x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ф
h
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_269432

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_270801

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
║
M
1__inference_max_pooling2d_67_layer_call_fn_270732

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_269508Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
М
А
E__inference_conv2d_67_layer_call_and_return_conditional_losses_270665

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╔
Щ
)__inference_dense_49_layer_call_fn_270849

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_269698p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┌F
∙

I__inference_sequential_22_layer_call_and_return_conditional_losses_270136
conv2d_66_input*
conv2d_66_270074:@
conv2d_66_270076:@+
batch_normalization_66_270079:@+
batch_normalization_66_270081:@+
batch_normalization_66_270083:@+
batch_normalization_66_270085:@+
conv2d_67_270089:@А
conv2d_67_270091:	А,
batch_normalization_67_270094:	А,
batch_normalization_67_270096:	А,
batch_normalization_67_270098:	А,
batch_normalization_67_270100:	А+
conv2d_68_270104:А@
conv2d_68_270106:@+
batch_normalization_68_270109:@+
batch_normalization_68_270111:@+
batch_normalization_68_270113:@+
batch_normalization_68_270115:@#
dense_49_270120:
АА
dense_49_270122:	А"
dense_50_270126:	А
dense_50_270128:
identityИв.batch_normalization_66/StatefulPartitionedCallв.batch_normalization_67/StatefulPartitionedCallв.batch_normalization_68/StatefulPartitionedCallв!conv2d_66/StatefulPartitionedCallв!conv2d_67/StatefulPartitionedCallв!conv2d_68/StatefulPartitionedCallв dense_49/StatefulPartitionedCallв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpв dense_50/StatefulPartitionedCallЕ
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputconv2d_66_270074conv2d_66_270076*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_269605Ц
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_66_270079batch_normalization_66_270081batch_normalization_66_270083batch_normalization_66_270085*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_269381Б
 max_pooling2d_66/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_269432а
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_66/PartitionedCall:output:0conv2d_67_270089conv2d_67_270091*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_269632Ч
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_67_270094batch_normalization_67_270096batch_normalization_67_270098batch_normalization_67_270100*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_269457В
 max_pooling2d_67/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_269508Я
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_67/PartitionedCall:output:0conv2d_68_270104conv2d_68_270106*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_269659Ц
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_68_270109batch_normalization_68_270111batch_normalization_68_270113batch_normalization_68_270115*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_269533Б
 max_pooling2d_68/PartitionedCallPartitionedCall7batch_normalization_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_269584р
flatten_22/PartitionedCallPartitionedCall)max_pooling2d_68/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_22_layer_call_and_return_conditional_losses_269681О
 dense_49/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_49_270120dense_49_270122*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_269698р
dropout_27/PartitionedCallPartitionedCall)dense_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_269709Н
 dense_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0dense_50_270126dense_50_270128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_50_layer_call_and_return_conditional_losses_269722Г
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_49_270120* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┐
NoOpNoOp/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall:` \
/
_output_shapes
:         dd
)
_user_specified_nameconv2d_66_input
║
M
1__inference_max_pooling2d_68_layer_call_fn_270824

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_269584Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_270617

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ф
h
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_269508

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▌
d
F__inference_dropout_27_layer_call_and_return_conditional_losses_270879

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
хG
Х
I__inference_sequential_22_layer_call_and_return_conditional_losses_269975

inputs*
conv2d_66_269913:@
conv2d_66_269915:@+
batch_normalization_66_269918:@+
batch_normalization_66_269920:@+
batch_normalization_66_269922:@+
batch_normalization_66_269924:@+
conv2d_67_269928:@А
conv2d_67_269930:	А,
batch_normalization_67_269933:	А,
batch_normalization_67_269935:	А,
batch_normalization_67_269937:	А,
batch_normalization_67_269939:	А+
conv2d_68_269943:А@
conv2d_68_269945:@+
batch_normalization_68_269948:@+
batch_normalization_68_269950:@+
batch_normalization_68_269952:@+
batch_normalization_68_269954:@#
dense_49_269959:
АА
dense_49_269961:	А"
dense_50_269965:	А
dense_50_269967:
identityИв.batch_normalization_66/StatefulPartitionedCallв.batch_normalization_67/StatefulPartitionedCallв.batch_normalization_68/StatefulPartitionedCallв!conv2d_66/StatefulPartitionedCallв!conv2d_67/StatefulPartitionedCallв!conv2d_68/StatefulPartitionedCallв dense_49/StatefulPartitionedCallв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpв dense_50/StatefulPartitionedCallв"dropout_27/StatefulPartitionedCall№
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_66_269913conv2d_66_269915*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_269605Ф
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_66_269918batch_normalization_66_269920batch_normalization_66_269922batch_normalization_66_269924*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_269412Б
 max_pooling2d_66/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_269432а
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_66/PartitionedCall:output:0conv2d_67_269928conv2d_67_269930*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_269632Х
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_67_269933batch_normalization_67_269935batch_normalization_67_269937batch_normalization_67_269939*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_269488В
 max_pooling2d_67/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_269508Я
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_67/PartitionedCall:output:0conv2d_68_269943conv2d_68_269945*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_269659Ф
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_68_269948batch_normalization_68_269950batch_normalization_68_269952batch_normalization_68_269954*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_269564Б
 max_pooling2d_68/PartitionedCallPartitionedCall7batch_normalization_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_269584р
flatten_22/PartitionedCallPartitionedCall)max_pooling2d_68/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_22_layer_call_and_return_conditional_losses_269681О
 dense_49/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_49_269959dense_49_269961*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_269698Ё
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_269810Х
 dense_50/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0dense_50_269965dense_50_269967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_50_layer_call_and_return_conditional_losses_269722Г
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_49_269959* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ф
NoOpNoOp/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_50/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
║
M
1__inference_max_pooling2d_66_layer_call_fn_270640

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_269432Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
№	
e
F__inference_dropout_27_layer_call_and_return_conditional_losses_270891

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ч
┼
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_270727

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
┼
═
.__inference_sequential_22_layer_call_fn_270315

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А%

unknown_11:А@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
АА

unknown_18:	А

unknown_19:	А

unknown_20:
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_269733o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
┐F
Ё

I__inference_sequential_22_layer_call_and_return_conditional_losses_269733

inputs*
conv2d_66_269606:@
conv2d_66_269608:@+
batch_normalization_66_269611:@+
batch_normalization_66_269613:@+
batch_normalization_66_269615:@+
batch_normalization_66_269617:@+
conv2d_67_269633:@А
conv2d_67_269635:	А,
batch_normalization_67_269638:	А,
batch_normalization_67_269640:	А,
batch_normalization_67_269642:	А,
batch_normalization_67_269644:	А+
conv2d_68_269660:А@
conv2d_68_269662:@+
batch_normalization_68_269665:@+
batch_normalization_68_269667:@+
batch_normalization_68_269669:@+
batch_normalization_68_269671:@#
dense_49_269699:
АА
dense_49_269701:	А"
dense_50_269723:	А
dense_50_269725:
identityИв.batch_normalization_66/StatefulPartitionedCallв.batch_normalization_67/StatefulPartitionedCallв.batch_normalization_68/StatefulPartitionedCallв!conv2d_66/StatefulPartitionedCallв!conv2d_67/StatefulPartitionedCallв!conv2d_68/StatefulPartitionedCallв dense_49/StatefulPartitionedCallв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpв dense_50/StatefulPartitionedCall№
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_66_269606conv2d_66_269608*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_269605Ц
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_66_269611batch_normalization_66_269613batch_normalization_66_269615batch_normalization_66_269617*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_269381Б
 max_pooling2d_66/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_269432а
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_66/PartitionedCall:output:0conv2d_67_269633conv2d_67_269635*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_269632Ч
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_67_269638batch_normalization_67_269640batch_normalization_67_269642batch_normalization_67_269644*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_269457В
 max_pooling2d_67/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_269508Я
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_67/PartitionedCall:output:0conv2d_68_269660conv2d_68_269662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_269659Ц
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_68_269665batch_normalization_68_269667batch_normalization_68_269669batch_normalization_68_269671*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_269533Б
 max_pooling2d_68/PartitionedCallPartitionedCall7batch_normalization_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_269584р
flatten_22/PartitionedCallPartitionedCall)max_pooling2d_68/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_22_layer_call_and_return_conditional_losses_269681О
 dense_49/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_49_269699dense_49_269701*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_269698р
dropout_27/PartitionedCallPartitionedCall)dense_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_269709Н
 dense_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0dense_50_269723dense_50_269725*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_50_layer_call_and_return_conditional_losses_269722Г
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_49_269699* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ┐
NoOpNoOp/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_50/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
ў
d
+__inference_dropout_27_layer_call_fn_270874

inputs
identityИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_269810p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ф
h
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_270645

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
о
╠
$__inference_signature_wrapper_270262
conv2d_66_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А%

unknown_11:А@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
АА

unknown_18:	А

unknown_19:	А

unknown_20:
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_269359o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         dd
)
_user_specified_nameconv2d_66_input
Щ	
╓
7__inference_batch_normalization_67_layer_call_fn_270691

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_269488К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
С	
╥
7__inference_batch_normalization_66_layer_call_fn_270599

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_269412Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Л 
д)
"__inference__traced_restore_271331
file_prefix;
!assignvariableop_conv2d_66_kernel:@/
!assignvariableop_1_conv2d_66_bias:@=
/assignvariableop_2_batch_normalization_66_gamma:@<
.assignvariableop_3_batch_normalization_66_beta:@C
5assignvariableop_4_batch_normalization_66_moving_mean:@G
9assignvariableop_5_batch_normalization_66_moving_variance:@>
#assignvariableop_6_conv2d_67_kernel:@А0
!assignvariableop_7_conv2d_67_bias:	А>
/assignvariableop_8_batch_normalization_67_gamma:	А=
.assignvariableop_9_batch_normalization_67_beta:	АE
6assignvariableop_10_batch_normalization_67_moving_mean:	АI
:assignvariableop_11_batch_normalization_67_moving_variance:	А?
$assignvariableop_12_conv2d_68_kernel:А@0
"assignvariableop_13_conv2d_68_bias:@>
0assignvariableop_14_batch_normalization_68_gamma:@=
/assignvariableop_15_batch_normalization_68_beta:@D
6assignvariableop_16_batch_normalization_68_moving_mean:@H
:assignvariableop_17_batch_normalization_68_moving_variance:@7
#assignvariableop_18_dense_49_kernel:
АА0
!assignvariableop_19_dense_49_bias:	А6
#assignvariableop_20_dense_50_kernel:	А/
!assignvariableop_21_dense_50_bias:'
assignvariableop_22_adam_iter:	 )
assignvariableop_23_adam_beta_1: )
assignvariableop_24_adam_beta_2: (
assignvariableop_25_adam_decay: 0
&assignvariableop_26_adam_learning_rate: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: #
assignvariableop_29_total: #
assignvariableop_30_count: E
+assignvariableop_31_adam_conv2d_66_kernel_m:@7
)assignvariableop_32_adam_conv2d_66_bias_m:@E
7assignvariableop_33_adam_batch_normalization_66_gamma_m:@D
6assignvariableop_34_adam_batch_normalization_66_beta_m:@F
+assignvariableop_35_adam_conv2d_67_kernel_m:@А8
)assignvariableop_36_adam_conv2d_67_bias_m:	АF
7assignvariableop_37_adam_batch_normalization_67_gamma_m:	АE
6assignvariableop_38_adam_batch_normalization_67_beta_m:	АF
+assignvariableop_39_adam_conv2d_68_kernel_m:А@7
)assignvariableop_40_adam_conv2d_68_bias_m:@E
7assignvariableop_41_adam_batch_normalization_68_gamma_m:@D
6assignvariableop_42_adam_batch_normalization_68_beta_m:@>
*assignvariableop_43_adam_dense_49_kernel_m:
АА7
(assignvariableop_44_adam_dense_49_bias_m:	А=
*assignvariableop_45_adam_dense_50_kernel_m:	А6
(assignvariableop_46_adam_dense_50_bias_m:E
+assignvariableop_47_adam_conv2d_66_kernel_v:@7
)assignvariableop_48_adam_conv2d_66_bias_v:@E
7assignvariableop_49_adam_batch_normalization_66_gamma_v:@D
6assignvariableop_50_adam_batch_normalization_66_beta_v:@F
+assignvariableop_51_adam_conv2d_67_kernel_v:@А8
)assignvariableop_52_adam_conv2d_67_bias_v:	АF
7assignvariableop_53_adam_batch_normalization_67_gamma_v:	АE
6assignvariableop_54_adam_batch_normalization_67_beta_v:	АF
+assignvariableop_55_adam_conv2d_68_kernel_v:А@7
)assignvariableop_56_adam_conv2d_68_bias_v:@E
7assignvariableop_57_adam_batch_normalization_68_gamma_v:@D
6assignvariableop_58_adam_batch_normalization_68_beta_v:@>
*assignvariableop_59_adam_dense_49_kernel_v:
АА7
(assignvariableop_60_adam_dense_49_bias_v:	А=
*assignvariableop_61_adam_dense_50_kernel_v:	А6
(assignvariableop_62_adam_dense_50_bias_v:
identity_64ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Л#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*▒"
valueз"Bд"@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHє
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Х
valueЛBИ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B с
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_66_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_66_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_66_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_66_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_66_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_66_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_67_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_67_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_67_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_67_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_67_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_67_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_68_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_68_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_68_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_68_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_68_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_68_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_49_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_49_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_50_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_50_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_66_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_66_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_batch_normalization_66_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_66_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_67_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_67_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_67_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_67_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_68_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_68_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_68_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_68_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_49_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_49_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_50_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_50_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_66_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_66_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_66_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_66_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_67_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_67_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_67_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_67_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_68_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_68_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_68_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_68_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_49_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_49_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_50_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_50_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╣
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: ж
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_64Identity_64:output:0*Х
_input_shapesГ
А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
═
Э
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_269381

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
№	
e
F__inference_dropout_27_layer_call_and_return_conditional_losses_269810

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_269564

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_269412

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
АH
Ю
I__inference_sequential_22_layer_call_and_return_conditional_losses_270201
conv2d_66_input*
conv2d_66_270139:@
conv2d_66_270141:@+
batch_normalization_66_270144:@+
batch_normalization_66_270146:@+
batch_normalization_66_270148:@+
batch_normalization_66_270150:@+
conv2d_67_270154:@А
conv2d_67_270156:	А,
batch_normalization_67_270159:	А,
batch_normalization_67_270161:	А,
batch_normalization_67_270163:	А,
batch_normalization_67_270165:	А+
conv2d_68_270169:А@
conv2d_68_270171:@+
batch_normalization_68_270174:@+
batch_normalization_68_270176:@+
batch_normalization_68_270178:@+
batch_normalization_68_270180:@#
dense_49_270185:
АА
dense_49_270187:	А"
dense_50_270191:	А
dense_50_270193:
identityИв.batch_normalization_66/StatefulPartitionedCallв.batch_normalization_67/StatefulPartitionedCallв.batch_normalization_68/StatefulPartitionedCallв!conv2d_66/StatefulPartitionedCallв!conv2d_67/StatefulPartitionedCallв!conv2d_68/StatefulPartitionedCallв dense_49/StatefulPartitionedCallв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpв dense_50/StatefulPartitionedCallв"dropout_27/StatefulPartitionedCallЕ
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputconv2d_66_270139conv2d_66_270141*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_269605Ф
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_66_270144batch_normalization_66_270146batch_normalization_66_270148batch_normalization_66_270150*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_269412Б
 max_pooling2d_66/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_269432а
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_66/PartitionedCall:output:0conv2d_67_270154conv2d_67_270156*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_269632Х
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_67_270159batch_normalization_67_270161batch_normalization_67_270163batch_normalization_67_270165*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_269488В
 max_pooling2d_67/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_269508Я
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_67/PartitionedCall:output:0conv2d_68_270169conv2d_68_270171*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_269659Ф
.batch_normalization_68/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_68_270174batch_normalization_68_270176batch_normalization_68_270178batch_normalization_68_270180*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_269564Б
 max_pooling2d_68/PartitionedCallPartitionedCall7batch_normalization_68/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_269584р
flatten_22/PartitionedCallPartitionedCall)max_pooling2d_68/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_22_layer_call_and_return_conditional_losses_269681О
 dense_49/StatefulPartitionedCallStatefulPartitionedCall#flatten_22/PartitionedCall:output:0dense_49_270185dense_49_270187*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_49_layer_call_and_return_conditional_losses_269698Ё
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_27_layer_call_and_return_conditional_losses_269810Х
 dense_50/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0dense_50_270191dense_50_270193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_50_layer_call_and_return_conditional_losses_269722Г
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_49_270185* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_50/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ф
NoOpNoOp/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall/^batch_normalization_68/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_50/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2`
.batch_normalization_68/StatefulPartitionedCall.batch_normalization_68/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall:` \
/
_output_shapes
:         dd
)
_user_specified_nameconv2d_66_input
┌
╓
.__inference_sequential_22_layer_call_fn_270071
conv2d_66_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А%

unknown_11:А@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
АА

unknown_18:	А

unknown_19:	А

unknown_20:
identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_269975o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         dd
)
_user_specified_nameconv2d_66_input
г
м
D__inference_dense_49_layer_call_and_return_conditional_losses_270864

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         АТ
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
д

Ў
D__inference_dense_50_layer_call_and_return_conditional_losses_270911

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_66_layer_call_and_return_conditional_losses_270573

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_269533

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
р
╓
.__inference_sequential_22_layer_call_fn_269780
conv2d_66_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@А
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А%

unknown_11:А@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
АА

unknown_18:	А

unknown_19:	А

unknown_20:
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_269733o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         dd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         dd
)
_user_specified_nameconv2d_66_input
▌
б
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_269457

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ч
┼
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_269488

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ы	
╓
7__inference_batch_normalization_67_layer_call_fn_270678

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_269457К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
г
м
D__inference_dense_49_layer_call_and_return_conditional_losses_269698

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         АТ
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_49/kernel/Regularizer/L2LossL2Loss9dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_49/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_49/kernel/Regularizer/mulMul*dense_49/kernel/Regularizer/mul/x:output:0+dense_49/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_49/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp1dense_49/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
С	
╥
7__inference_batch_normalization_68_layer_call_fn_270783

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_269564Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_66_layer_call_and_return_conditional_losses_269605

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_270635

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┼
Ч
)__inference_dense_50_layer_call_fn_270900

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_50_layer_call_and_return_conditional_losses_269722o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*├
serving_defaultп
S
conv2d_66_input@
!serving_default_conv2d_66_input:0         dd<
dense_500
StatefulPartitionedCall:0         tensorflow/serving/predict:Ес
т
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
▌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
ъ
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&axis
	'gamma
(beta
)moving_mean
*moving_variance"
_tf_keras_layer
е
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op"
_tf_keras_layer
ъ
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance"
_tf_keras_layer
е
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias
 S_jit_compiled_convolution_op"
_tf_keras_layer
ъ
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance"
_tf_keras_layer
е
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
е
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias"
_tf_keras_layer
╝
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
y_random_generator"
_tf_keras_layer
╜
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
Аkernel
	Бbias"
_tf_keras_layer
╚
0
1
'2
(3
)4
*5
76
87
A8
B9
C10
D11
Q12
R13
[14
\15
]16
^17
q18
r19
А20
Б21"
trackable_list_wrapper
Ш
0
1
'2
(3
74
85
A6
B7
Q8
R9
[10
\11
q12
r13
А14
Б15"
trackable_list_wrapper
(
В0"
trackable_list_wrapper
╧
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ї
Иtrace_0
Йtrace_1
Кtrace_2
Лtrace_32В
.__inference_sequential_22_layer_call_fn_269780
.__inference_sequential_22_layer_call_fn_270315
.__inference_sequential_22_layer_call_fn_270364
.__inference_sequential_22_layer_call_fn_270071┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zИtrace_0zЙtrace_1zКtrace_2zЛtrace_3
с
Мtrace_0
Нtrace_1
Оtrace_2
Пtrace_32ю
I__inference_sequential_22_layer_call_and_return_conditional_losses_270455
I__inference_sequential_22_layer_call_and_return_conditional_losses_270553
I__inference_sequential_22_layer_call_and_return_conditional_losses_270136
I__inference_sequential_22_layer_call_and_return_conditional_losses_270201┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0zНtrace_1zОtrace_2zПtrace_3
╘B╤
!__inference__wrapped_model_269359conv2d_66_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ь
	Рiter
Сbeta_1
Тbeta_2

Уdecay
Фlearning_ratemЕmЖ'mЗ(mИ7mЙ8mКAmЛBmМQmНRmО[mП\mРqmСrmТ	АmУ	БmФvХvЦ'vЧ(vШ7vЩ8vЪAvЫBvЬQvЭRvЮ[vЯ\vаqvбrvв	Аvг	Бvд"
	optimizer
-
Хserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ё
Ыtrace_02╤
*__inference_conv2d_66_layer_call_fn_270562в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
Л
Ьtrace_02ь
E__inference_conv2d_66_layer_call_and_return_conditional_losses_270573в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЬtrace_0
*:(@2conv2d_66/kernel
:@2conv2d_66/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
'0
(1
)2
*3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
у
вtrace_0
гtrace_12и
7__inference_batch_normalization_66_layer_call_fn_270586
7__inference_batch_normalization_66_layer_call_fn_270599│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0zгtrace_1
Щ
дtrace_0
еtrace_12▐
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_270617
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_270635│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zдtrace_0zеtrace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_66/gamma
):'@2batch_normalization_66/beta
2:0@ (2"batch_normalization_66/moving_mean
6:4@ (2&batch_normalization_66/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
ў
лtrace_02╪
1__inference_max_pooling2d_66_layer_call_fn_270640в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0
Т
мtrace_02є
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_270645в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
Ё
▓trace_02╤
*__inference_conv2d_67_layer_call_fn_270654в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▓trace_0
Л
│trace_02ь
E__inference_conv2d_67_layer_call_and_return_conditional_losses_270665в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0
+:)@А2conv2d_67/kernel
:А2conv2d_67/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
A0
B1
C2
D3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
у
╣trace_0
║trace_12и
7__inference_batch_normalization_67_layer_call_fn_270678
7__inference_batch_normalization_67_layer_call_fn_270691│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╣trace_0z║trace_1
Щ
╗trace_0
╝trace_12▐
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_270709
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_270727│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0z╝trace_1
 "
trackable_list_wrapper
+:)А2batch_normalization_67/gamma
*:(А2batch_normalization_67/beta
3:1А (2"batch_normalization_67/moving_mean
7:5А (2&batch_normalization_67/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
ў
┬trace_02╪
1__inference_max_pooling2d_67_layer_call_fn_270732в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0
Т
├trace_02є
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_270737в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Ё
╔trace_02╤
*__inference_conv2d_68_layer_call_fn_270746в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╔trace_0
Л
╩trace_02ь
E__inference_conv2d_68_layer_call_and_return_conditional_losses_270757в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
+:)А@2conv2d_68/kernel
:@2conv2d_68/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
<
[0
\1
]2
^3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
у
╨trace_0
╤trace_12и
7__inference_batch_normalization_68_layer_call_fn_270770
7__inference_batch_normalization_68_layer_call_fn_270783│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╨trace_0z╤trace_1
Щ
╥trace_0
╙trace_12▐
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_270801
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_270819│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0z╙trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_68/gamma
):'@2batch_normalization_68/beta
2:0@ (2"batch_normalization_68/moving_mean
6:4@ (2&batch_normalization_68/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
ў
┘trace_02╪
1__inference_max_pooling2d_68_layer_call_fn_270824в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
Т
┌trace_02є
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_270829в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┌trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
ё
рtrace_02╥
+__inference_flatten_22_layer_call_fn_270834в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zрtrace_0
М
сtrace_02э
F__inference_flatten_22_layer_call_and_return_conditional_losses_270840в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
(
В0"
trackable_list_wrapper
▓
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
я
чtrace_02╨
)__inference_dense_49_layer_call_fn_270849в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0
К
шtrace_02ы
D__inference_dense_49_layer_call_and_return_conditional_losses_270864в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zшtrace_0
#:!
АА2dense_49/kernel
:А2dense_49/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
╦
юtrace_0
яtrace_12Р
+__inference_dropout_27_layer_call_fn_270869
+__inference_dropout_27_layer_call_fn_270874│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zюtrace_0zяtrace_1
Б
Ёtrace_0
ёtrace_12╞
F__inference_dropout_27_layer_call_and_return_conditional_losses_270879
F__inference_dropout_27_layer_call_and_return_conditional_losses_270891│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0zёtrace_1
"
_generic_user_object
0
А0
Б1"
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
я
ўtrace_02╨
)__inference_dense_50_layer_call_fn_270900в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
К
°trace_02ы
D__inference_dense_50_layer_call_and_return_conditional_losses_270911в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0
": 	А2dense_50/kernel
:2dense_50/bias
╧
∙trace_02░
__inference_loss_fn_0_270920П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z∙trace_0
J
)0
*1
C2
D3
]4
^5"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
0
·0
√1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ИBЕ
.__inference_sequential_22_layer_call_fn_269780conv2d_66_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
.__inference_sequential_22_layer_call_fn_270315inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
.__inference_sequential_22_layer_call_fn_270364inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
.__inference_sequential_22_layer_call_fn_270071conv2d_66_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
I__inference_sequential_22_layer_call_and_return_conditional_losses_270455inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
I__inference_sequential_22_layer_call_and_return_conditional_losses_270553inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
гBа
I__inference_sequential_22_layer_call_and_return_conditional_losses_270136conv2d_66_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
гBа
I__inference_sequential_22_layer_call_and_return_conditional_losses_270201conv2d_66_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╙B╨
$__inference_signature_wrapper_270262conv2d_66_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_conv2d_66_layer_call_fn_270562inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_66_layer_call_and_return_conditional_losses_270573inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_66_layer_call_fn_270586inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_66_layer_call_fn_270599inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_270617inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_270635inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_max_pooling2d_66_layer_call_fn_270640inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_270645inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_conv2d_67_layer_call_fn_270654inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_67_layer_call_and_return_conditional_losses_270665inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_67_layer_call_fn_270678inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_67_layer_call_fn_270691inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_270709inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_270727inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_max_pooling2d_67_layer_call_fn_270732inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_270737inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_conv2d_68_layer_call_fn_270746inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_conv2d_68_layer_call_and_return_conditional_losses_270757inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_68_layer_call_fn_270770inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_68_layer_call_fn_270783inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_270801inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_270819inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_max_pooling2d_68_layer_call_fn_270824inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_270829inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_flatten_22_layer_call_fn_270834inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_flatten_22_layer_call_and_return_conditional_losses_270840inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
В0"
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_dense_49_layer_call_fn_270849inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_dense_49_layer_call_and_return_conditional_losses_270864inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBэ
+__inference_dropout_27_layer_call_fn_270869inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
+__inference_dropout_27_layer_call_fn_270874inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_27_layer_call_and_return_conditional_losses_270879inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЛBИ
F__inference_dropout_27_layer_call_and_return_conditional_losses_270891inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_dense_50_layer_call_fn_270900inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_dense_50_layer_call_and_return_conditional_losses_270911inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
│B░
__inference_loss_fn_0_270920"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
R
№	variables
¤	keras_api

■total

 count"
_tf_keras_metric
c
А	variables
Б	keras_api

Вtotal

Гcount
Д
_fn_kwargs"
_tf_keras_metric
0
■0
 1"
trackable_list_wrapper
.
№	variables"
_generic_user_object
:  (2total
:  (2count
0
В0
Г1"
trackable_list_wrapper
.
А	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-@2Adam/conv2d_66/kernel/m
!:@2Adam/conv2d_66/bias/m
/:-@2#Adam/batch_normalization_66/gamma/m
.:,@2"Adam/batch_normalization_66/beta/m
0:.@А2Adam/conv2d_67/kernel/m
": А2Adam/conv2d_67/bias/m
0:.А2#Adam/batch_normalization_67/gamma/m
/:-А2"Adam/batch_normalization_67/beta/m
0:.А@2Adam/conv2d_68/kernel/m
!:@2Adam/conv2d_68/bias/m
/:-@2#Adam/batch_normalization_68/gamma/m
.:,@2"Adam/batch_normalization_68/beta/m
(:&
АА2Adam/dense_49/kernel/m
!:А2Adam/dense_49/bias/m
':%	А2Adam/dense_50/kernel/m
 :2Adam/dense_50/bias/m
/:-@2Adam/conv2d_66/kernel/v
!:@2Adam/conv2d_66/bias/v
/:-@2#Adam/batch_normalization_66/gamma/v
.:,@2"Adam/batch_normalization_66/beta/v
0:.@А2Adam/conv2d_67/kernel/v
": А2Adam/conv2d_67/bias/v
0:.А2#Adam/batch_normalization_67/gamma/v
/:-А2"Adam/batch_normalization_67/beta/v
0:.А@2Adam/conv2d_68/kernel/v
!:@2Adam/conv2d_68/bias/v
/:-@2#Adam/batch_normalization_68/gamma/v
.:,@2"Adam/batch_normalization_68/beta/v
(:&
АА2Adam/dense_49/kernel/v
!:А2Adam/dense_49/bias/v
':%	А2Adam/dense_50/kernel/v
 :2Adam/dense_50/bias/v╖
!__inference__wrapped_model_269359С'()*78ABCDQR[\]^qrАБ@в=
6в3
1К.
conv2d_66_input         dd
к "3к0
.
dense_50"К
dense_50         э
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_270617Ц'()*MвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ э
R__inference_batch_normalization_66_layer_call_and_return_conditional_losses_270635Ц'()*MвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ┼
7__inference_batch_normalization_66_layer_call_fn_270586Й'()*MвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @┼
7__inference_batch_normalization_66_layer_call_fn_270599Й'()*MвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @я
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_270709ШABCDNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ я
R__inference_batch_normalization_67_layer_call_and_return_conditional_losses_270727ШABCDNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ╟
7__inference_batch_normalization_67_layer_call_fn_270678ЛABCDNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╟
7__inference_batch_normalization_67_layer_call_fn_270691ЛABCDNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           Аэ
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_270801Ц[\]^MвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ э
R__inference_batch_normalization_68_layer_call_and_return_conditional_losses_270819Ц[\]^MвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ┼
7__inference_batch_normalization_68_layer_call_fn_270770Й[\]^MвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @┼
7__inference_batch_normalization_68_layer_call_fn_270783Й[\]^MвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @╡
E__inference_conv2d_66_layer_call_and_return_conditional_losses_270573l7в4
-в*
(К%
inputs         dd
к "-в*
#К 
0         @
Ъ Н
*__inference_conv2d_66_layer_call_fn_270562_7в4
-в*
(К%
inputs         dd
к " К         @╢
E__inference_conv2d_67_layer_call_and_return_conditional_losses_270665m787в4
-в*
(К%
inputs         @
к ".в+
$К!
0         А
Ъ О
*__inference_conv2d_67_layer_call_fn_270654`787в4
-в*
(К%
inputs         @
к "!К         А╢
E__inference_conv2d_68_layer_call_and_return_conditional_losses_270757mQR8в5
.в+
)К&
inputs         А
к "-в*
#К 
0         @
Ъ О
*__inference_conv2d_68_layer_call_fn_270746`QR8в5
.в+
)К&
inputs         А
к " К         @ж
D__inference_dense_49_layer_call_and_return_conditional_losses_270864^qr0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ ~
)__inference_dense_49_layer_call_fn_270849Qqr0в-
&в#
!К
inputs         А
к "К         Аз
D__inference_dense_50_layer_call_and_return_conditional_losses_270911_АБ0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ 
)__inference_dense_50_layer_call_fn_270900RАБ0в-
&в#
!К
inputs         А
к "К         и
F__inference_dropout_27_layer_call_and_return_conditional_losses_270879^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ и
F__inference_dropout_27_layer_call_and_return_conditional_losses_270891^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ А
+__inference_dropout_27_layer_call_fn_270869Q4в1
*в'
!К
inputs         А
p 
к "К         АА
+__inference_dropout_27_layer_call_fn_270874Q4в1
*в'
!К
inputs         А
p
к "К         Ал
F__inference_flatten_22_layer_call_and_return_conditional_losses_270840a7в4
-в*
(К%
inputs         @
к "&в#
К
0         А
Ъ Г
+__inference_flatten_22_layer_call_fn_270834T7в4
-в*
(К%
inputs         @
к "К         А;
__inference_loss_fn_0_270920qв

в 
к "К я
L__inference_max_pooling2d_66_layer_call_and_return_conditional_losses_270645ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_66_layer_call_fn_270640СRвO
HвE
CК@
inputs4                                    
к ";К84                                    я
L__inference_max_pooling2d_67_layer_call_and_return_conditional_losses_270737ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_67_layer_call_fn_270732СRвO
HвE
CК@
inputs4                                    
к ";К84                                    я
L__inference_max_pooling2d_68_layer_call_and_return_conditional_losses_270829ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_68_layer_call_fn_270824СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ┘
I__inference_sequential_22_layer_call_and_return_conditional_losses_270136Л'()*78ABCDQR[\]^qrАБHвE
>в;
1К.
conv2d_66_input         dd
p 

 
к "%в"
К
0         
Ъ ┘
I__inference_sequential_22_layer_call_and_return_conditional_losses_270201Л'()*78ABCDQR[\]^qrАБHвE
>в;
1К.
conv2d_66_input         dd
p

 
к "%в"
К
0         
Ъ ╨
I__inference_sequential_22_layer_call_and_return_conditional_losses_270455В'()*78ABCDQR[\]^qrАБ?в<
5в2
(К%
inputs         dd
p 

 
к "%в"
К
0         
Ъ ╨
I__inference_sequential_22_layer_call_and_return_conditional_losses_270553В'()*78ABCDQR[\]^qrАБ?в<
5в2
(К%
inputs         dd
p

 
к "%в"
К
0         
Ъ ░
.__inference_sequential_22_layer_call_fn_269780~'()*78ABCDQR[\]^qrАБHвE
>в;
1К.
conv2d_66_input         dd
p 

 
к "К         ░
.__inference_sequential_22_layer_call_fn_270071~'()*78ABCDQR[\]^qrАБHвE
>в;
1К.
conv2d_66_input         dd
p

 
к "К         з
.__inference_sequential_22_layer_call_fn_270315u'()*78ABCDQR[\]^qrАБ?в<
5в2
(К%
inputs         dd
p 

 
к "К         з
.__inference_sequential_22_layer_call_fn_270364u'()*78ABCDQR[\]^qrАБ?в<
5в2
(К%
inputs         dd
p

 
к "К         ═
$__inference_signature_wrapper_270262д'()*78ABCDQR[\]^qrАБSвP
в 
IкF
D
conv2d_66_input1К.
conv2d_66_input         dd"3к0
.
dense_50"К
dense_50         