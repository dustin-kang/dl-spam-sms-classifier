ø
î
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-0-g3f878cff5b68Óú

embedding_7/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ÙE *'
shared_nameembedding_7/embeddings

*embedding_7/embeddings/Read/ReadVariableOpReadVariableOpembedding_7/embeddings*
_output_shapes
:	ÙE *
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

: *
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
¦
%simple_rnn_7/simple_rnn_cell_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *6
shared_name'%simple_rnn_7/simple_rnn_cell_7/kernel

9simple_rnn_7/simple_rnn_cell_7/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_7/simple_rnn_cell_7/kernel*
_output_shapes

:  *
dtype0
º
/simple_rnn_7/simple_rnn_cell_7/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *@
shared_name1/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel
³
Csimple_rnn_7/simple_rnn_cell_7/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel*
_output_shapes

:  *
dtype0

#simple_rnn_7/simple_rnn_cell_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#simple_rnn_7/simple_rnn_cell_7/bias

7simple_rnn_7/simple_rnn_cell_7/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_7/simple_rnn_cell_7/bias*
_output_shapes
: *
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
¡
"RMSprop/embedding_7/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ÙE *3
shared_name$"RMSprop/embedding_7/embeddings/rms

6RMSprop/embedding_7/embeddings/rms/Read/ReadVariableOpReadVariableOp"RMSprop/embedding_7/embeddings/rms*
_output_shapes
:	ÙE *
dtype0

RMSprop/dense_7/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameRMSprop/dense_7/kernel/rms

.RMSprop/dense_7/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_7/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/dense_7/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_7/bias/rms

,RMSprop/dense_7/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_7/bias/rms*
_output_shapes
:*
dtype0
¾
1RMSprop/simple_rnn_7/simple_rnn_cell_7/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *B
shared_name31RMSprop/simple_rnn_7/simple_rnn_cell_7/kernel/rms
·
ERMSprop/simple_rnn_7/simple_rnn_cell_7/kernel/rms/Read/ReadVariableOpReadVariableOp1RMSprop/simple_rnn_7/simple_rnn_cell_7/kernel/rms*
_output_shapes

:  *
dtype0
Ò
;RMSprop/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *L
shared_name=;RMSprop/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/rms
Ë
ORMSprop/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp;RMSprop/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/rms*
_output_shapes

:  *
dtype0
¶
/RMSprop/simple_rnn_7/simple_rnn_cell_7/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/RMSprop/simple_rnn_7/simple_rnn_cell_7/bias/rms
¯
CRMSprop/simple_rnn_7/simple_rnn_cell_7/bias/rms/Read/ReadVariableOpReadVariableOp/RMSprop/simple_rnn_7/simple_rnn_cell_7/bias/rms*
_output_shapes
: *
dtype0

NoOpNoOp
È,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*,
valueù+Bö+ Bï+
Á
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
ª
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*

$iter
	%decay
&learning_rate
'momentum
(rho	rmsY	rmsZ	rms[	)rms\	*rms]	+rms^*
.
0
)1
*2
+3
4
5*
.
0
)1
*2
+3
4
5*
* 
°
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

1serving_default* 
jd
VARIABLE_VALUEembedding_7/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
Ó

)kernel
*recurrent_kernel
+bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;_random_generator
<__call__
*=&call_and_return_all_conditional_losses*
* 

)0
*1
+2*

)0
*1
+2*
* 


>states
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_7/simple_rnn_cell_7/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_7/simple_rnn_cell_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

I0
J1*
* 
* 
* 
* 
* 
* 
* 
* 

)0
*1
+2*

)0
*1
+2*
* 

Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
7	variables
8trainable_variables
9regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ptotal
	Qcount
R	variables
S	keras_api*
H
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

R	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

T0
U1*

W	variables*

VARIABLE_VALUE"RMSprop/embedding_7/embeddings/rmsXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_7/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUERMSprop/dense_7/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE1RMSprop/simple_rnn_7/simple_rnn_cell_7/kernel/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE;RMSprop/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE/RMSprop/simple_rnn_7/simple_rnn_cell_7/bias/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

!serving_default_embedding_7_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ú
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_7_inputembedding_7/embeddings%simple_rnn_7/simple_rnn_cell_7/kernel#simple_rnn_7/simple_rnn_cell_7/bias/simple_rnn_7/simple_rnn_cell_7/recurrent_kerneldense_7/kerneldense_7/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_94243
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ô	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_7/embeddings/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp9simple_rnn_7/simple_rnn_cell_7/kernel/Read/ReadVariableOpCsimple_rnn_7/simple_rnn_cell_7/recurrent_kernel/Read/ReadVariableOp7simple_rnn_7/simple_rnn_cell_7/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6RMSprop/embedding_7/embeddings/rms/Read/ReadVariableOp.RMSprop/dense_7/kernel/rms/Read/ReadVariableOp,RMSprop/dense_7/bias/rms/Read/ReadVariableOpERMSprop/simple_rnn_7/simple_rnn_cell_7/kernel/rms/Read/ReadVariableOpORMSprop/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/rms/Read/ReadVariableOpCRMSprop/simple_rnn_7/simple_rnn_cell_7/bias/rms/Read/ReadVariableOpConst*"
Tin
2	*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_94904
Ë
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_7/embeddingsdense_7/kerneldense_7/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rho%simple_rnn_7/simple_rnn_cell_7/kernel/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel#simple_rnn_7/simple_rnn_cell_7/biastotalcounttotal_1count_1"RMSprop/embedding_7/embeddings/rmsRMSprop/dense_7/kernel/rmsRMSprop/dense_7/bias/rms1RMSprop/simple_rnn_7/simple_rnn_cell_7/kernel/rms;RMSprop/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/rms/RMSprop/simple_rnn_7/simple_rnn_cell_7/bias/rms*!
Tin
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_94977¦
Ð3


__inference__traced_save_94904
file_prefix5
1savev2_embedding_7_embeddings_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableopD
@savev2_simple_rnn_7_simple_rnn_cell_7_kernel_read_readvariableopN
Jsavev2_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_7_simple_rnn_cell_7_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_rmsprop_embedding_7_embeddings_rms_read_readvariableop9
5savev2_rmsprop_dense_7_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_7_bias_rms_read_readvariableopP
Lsavev2_rmsprop_simple_rnn_7_simple_rnn_cell_7_kernel_rms_read_readvariableopZ
Vsavev2_rmsprop_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_rms_read_readvariableopN
Jsavev2_rmsprop_simple_rnn_7_simple_rnn_cell_7_bias_rms_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¾

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ç	
valueÝ	BÚ	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_7_embeddings_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop@savev2_simple_rnn_7_simple_rnn_cell_7_kernel_read_readvariableopJsavev2_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_read_readvariableop>savev2_simple_rnn_7_simple_rnn_cell_7_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_rmsprop_embedding_7_embeddings_rms_read_readvariableop5savev2_rmsprop_dense_7_kernel_rms_read_readvariableop3savev2_rmsprop_dense_7_bias_rms_read_readvariableopLsavev2_rmsprop_simple_rnn_7_simple_rnn_cell_7_kernel_rms_read_readvariableopVsavev2_rmsprop_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_rms_read_readvariableopJsavev2_rmsprop_simple_rnn_7_simple_rnn_cell_7_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*
_input_shapes
: :	ÙE : :: : : : : :  :  : : : : : :	ÙE : ::  :  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ÙE :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

:  :$
 

_output_shapes

:  : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ÙE :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:  :$ 

_output_shapes

:  : 

_output_shapes
: :

_output_shapes
: 
ß	
£
F__inference_embedding_7_layer_call_and_return_conditional_losses_94260

inputs)
embedding_lookup_94254:	ÙE 
identity¢embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
embedding_lookupResourceGatherembedding_lookup_94254Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/94254*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
dtype0ª
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/94254*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ=
º
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93817

inputsB
0simple_rnn_cell_7_matmul_readvariableop_resource:  ?
1simple_rnn_cell_7_biasadd_readvariableop_resource: D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:  
identity¢(simple_rnn_cell_7/BiasAdd/ReadVariableOp¢'simple_rnn_cell_7/MatMul/ReadVariableOp¢)simple_rnn_cell_7/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¬
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMul_1MatMulzeros:output:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ö
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_93751*
condR
while_cond_93750*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ=
º
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94736

inputsB
0simple_rnn_cell_7_matmul_readvariableop_resource:  ?
1simple_rnn_cell_7_biasadd_readvariableop_resource: D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:  
identity¢(simple_rnn_cell_7/BiasAdd/ReadVariableOp¢'simple_rnn_cell_7/MatMul/ReadVariableOp¢)simple_rnn_cell_7/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¬
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMul_1MatMulzeros:output:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ö
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_94670*
condR
while_cond_94669*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ=
º
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94628

inputsB
0simple_rnn_cell_7_matmul_readvariableop_resource:  ?
1simple_rnn_cell_7_biasadd_readvariableop_resource: D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:  
identity¢(simple_rnn_cell_7/BiasAdd/ReadVariableOp¢'simple_rnn_cell_7/MatMul/ReadVariableOp¢)simple_rnn_cell_7/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¬
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMul_1MatMulzeros:output:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ö
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_94562*
condR
while_cond_94561*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ=
º
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93645

inputsB
0simple_rnn_cell_7_matmul_readvariableop_resource:  ?
1simple_rnn_cell_7_biasadd_readvariableop_resource: D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:  
identity¢(simple_rnn_cell_7/BiasAdd/ReadVariableOp¢'simple_rnn_cell_7/MatMul/ReadVariableOp¢)simple_rnn_cell_7/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¬
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMul_1MatMulzeros:output:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ö
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_93579*
condR
while_cond_93578*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	

,__inference_sequential_7_layer_call_fn_93965

inputs
unknown:	ÙE 
	unknown_0:  
	unknown_1: 
	unknown_2:  
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_93671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
ç
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_93273

inputs

states0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
Õ
¥
while_cond_94561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_94561___redundant_placeholder03
/while_while_cond_94561___redundant_placeholder13
/while_while_cond_94561___redundant_placeholder23
/while_while_cond_94561___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Õ
¥
while_cond_94453
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_94453___redundant_placeholder03
/while_while_cond_94453___redundant_placeholder13
/while_while_cond_94453___redundant_placeholder23
/while_while_cond_94453___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Õ
¥
while_cond_94669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_94669___redundant_placeholder03
/while_while_cond_94669___redundant_placeholder13
/while_while_cond_94669___redundant_placeholder23
/while_while_cond_94669___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:

¶
,__inference_simple_rnn_7_layer_call_fn_94293

inputs
unknown:  
	unknown_0: 
	unknown_1:  
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93645o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
µ


simple_rnn_7_while_cond_940296
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_28
4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1M
Isimple_rnn_7_while_simple_rnn_7_while_cond_94029___redundant_placeholder0M
Isimple_rnn_7_while_simple_rnn_7_while_cond_94029___redundant_placeholder1M
Isimple_rnn_7_while_simple_rnn_7_while_cond_94029___redundant_placeholder2M
Isimple_rnn_7_while_simple_rnn_7_while_cond_94029___redundant_placeholder3
simple_rnn_7_while_identity

simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Õ
¥
while_cond_94345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_94345___redundant_placeholder03
/while_while_cond_94345___redundant_placeholder13
/while_while_cond_94345___redundant_placeholder23
/while_while_cond_94345___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
	

,__inference_sequential_7_layer_call_fn_93982

inputs
unknown:	ÙE 
	unknown_0:  
	unknown_1: 
	unknown_2:  
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_93872o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼,
Ç
while_body_93751
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource:  E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢-while/simple_rnn_cell_7/MatMul/ReadVariableOp¢/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¦
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0Ã
while/simple_rnn_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0¾
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ª
 while/simple_rnn_cell_7/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ É
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ}
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
¼,
Ç
while_body_93579
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource:  E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢-while/simple_rnn_cell_7/MatMul/ReadVariableOp¢/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¦
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0Ã
while/simple_rnn_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0¾
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ª
 while/simple_rnn_cell_7/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ É
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ}
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ë

+__inference_embedding_7_layer_call_fn_94250

inputs
unknown:	ÙE 
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_93534|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ8
Í
simple_rnn_7_while_body_940306
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_25
1simple_rnn_7_while_simple_rnn_7_strided_slice_1_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  T
Fsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: Y
Gsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_43
/simple_rnn_7_while_simple_rnn_7_strided_slice_1o
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource:  R
Dsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource: W
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp¢<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ç
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0À
:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0ê
+simple_rnn_7/while/simple_rnn_cell_7/MatMulMatMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¾
;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0å
,simple_rnn_7/while/simple_rnn_cell_7/BiasAddBiasAdd5simple_rnn_7/while/simple_rnn_cell_7/MatMul:product:0Csimple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0Ñ
-simple_rnn_7/while/simple_rnn_cell_7/MatMul_1MatMul simple_rnn_7_while_placeholder_2Dsimple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó
(simple_rnn_7/while/simple_rnn_cell_7/addAddV25simple_rnn_7/while/simple_rnn_cell_7/BiasAdd:output:07simple_rnn_7/while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)simple_rnn_7/while/simple_rnn_cell_7/TanhTanh,simple_rnn_7/while/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ý
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder-simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒZ
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: À
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
simple_rnn_7/while/Identity_4Identity-simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0^simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
simple_rnn_7/while/NoOpNoOp<^simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp;^simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp=^simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"d
/simple_rnn_7_while_simple_rnn_7_strided_slice_11simple_rnn_7_while_simple_rnn_7_strided_slice_1_0"
Dsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resourceFsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"
Csimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resourceEsimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0"Ü
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2z
;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2x
:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp2|
<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ù=
¼
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94520
inputs_0B
0simple_rnn_cell_7_matmul_readvariableop_resource:  ?
1simple_rnn_cell_7_biasadd_readvariableop_resource: D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:  
identity¢(simple_rnn_cell_7/BiasAdd/ReadVariableOp¢'simple_rnn_cell_7/MatMul/ReadVariableOp¢)simple_rnn_cell_7/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¬
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMul_1MatMulzeros:output:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ö
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_94454*
condR
while_cond_94453*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
ü3

G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93349

inputs)
simple_rnn_cell_7_93274:  %
simple_rnn_cell_7_93276: )
simple_rnn_cell_7_93278:  
identity¢)simple_rnn_cell_7/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskã
)simple_rnn_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_7_93274simple_rnn_cell_7_93276simple_rnn_cell_7_93278*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_93273n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_7_93274simple_rnn_cell_7_93276simple_rnn_cell_7_93278*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_93286*
condR
while_cond_93285*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
NoOpNoOp*^simple_rnn_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2V
)simple_rnn_cell_7/StatefulPartitionedCall)simple_rnn_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¶
,__inference_simple_rnn_7_layer_call_fn_94304

inputs
unknown:  
	unknown_0: 
	unknown_1:  
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93817o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

é
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_94801

inputs
states_00
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
j
ª
 __inference__wrapped_model_93225
embedding_7_inputB
/sequential_7_embedding_7_embedding_lookup_93108:	ÙE \
Jsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource:  Y
Ksequential_7_simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource: ^
Lsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource:  E
3sequential_7_dense_7_matmul_readvariableop_resource: B
4sequential_7_dense_7_biasadd_readvariableop_resource:
identity¢+sequential_7/dense_7/BiasAdd/ReadVariableOp¢*sequential_7/dense_7/MatMul/ReadVariableOp¢)sequential_7/embedding_7/embedding_lookup¢Bsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢Asequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp¢Csequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp¢sequential_7/simple_rnn_7/while
sequential_7/embedding_7/CastCastembedding_7_input*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
)sequential_7/embedding_7/embedding_lookupResourceGather/sequential_7_embedding_7_embedding_lookup_93108!sequential_7/embedding_7/Cast:y:0*
Tindices0*B
_class8
64loc:@sequential_7/embedding_7/embedding_lookup/93108*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
dtype0õ
2sequential_7/embedding_7/embedding_lookup/IdentityIdentity2sequential_7/embedding_7/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential_7/embedding_7/embedding_lookup/93108*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¼
4sequential_7/embedding_7/embedding_lookup/Identity_1Identity;sequential_7/embedding_7/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
sequential_7/simple_rnn_7/ShapeShape=sequential_7/embedding_7/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:w
-sequential_7/simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_7/simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_7/simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'sequential_7/simple_rnn_7/strided_sliceStridedSlice(sequential_7/simple_rnn_7/Shape:output:06sequential_7/simple_rnn_7/strided_slice/stack:output:08sequential_7/simple_rnn_7/strided_slice/stack_1:output:08sequential_7/simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_7/simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Á
&sequential_7/simple_rnn_7/zeros/packedPack0sequential_7/simple_rnn_7/strided_slice:output:01sequential_7/simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_7/simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    º
sequential_7/simple_rnn_7/zerosFill/sequential_7/simple_rnn_7/zeros/packed:output:0.sequential_7/simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ }
(sequential_7/simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          á
#sequential_7/simple_rnn_7/transpose	Transpose=sequential_7/embedding_7/embedding_lookup/Identity_1:output:01sequential_7/simple_rnn_7/transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ x
!sequential_7/simple_rnn_7/Shape_1Shape'sequential_7/simple_rnn_7/transpose:y:0*
T0*
_output_shapes
:y
/sequential_7/simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_7/simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_7/simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)sequential_7/simple_rnn_7/strided_slice_1StridedSlice*sequential_7/simple_rnn_7/Shape_1:output:08sequential_7/simple_rnn_7/strided_slice_1/stack:output:0:sequential_7/simple_rnn_7/strided_slice_1/stack_1:output:0:sequential_7/simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5sequential_7/simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
'sequential_7/simple_rnn_7/TensorArrayV2TensorListReserve>sequential_7/simple_rnn_7/TensorArrayV2/element_shape:output:02sequential_7/simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ 
Osequential_7/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ®
Asequential_7/simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_7/simple_rnn_7/transpose:y:0Xsequential_7/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒy
/sequential_7/simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_7/simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_7/simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
)sequential_7/simple_rnn_7/strided_slice_2StridedSlice'sequential_7/simple_rnn_7/transpose:y:08sequential_7/simple_rnn_7/strided_slice_2/stack:output:0:sequential_7/simple_rnn_7/strided_slice_2/stack_1:output:0:sequential_7/simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskÌ
Asequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOpJsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0í
2sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMulMatMul2sequential_7/simple_rnn_7/strided_slice_2:output:0Isequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ê
Bsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOpKsequential_7_simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ú
3sequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAddBiasAdd<sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul:product:0Jsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ð
Csequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOpLsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0ç
4sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1MatMul(sequential_7/simple_rnn_7/zeros:output:0Ksequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ è
/sequential_7/simple_rnn_7/simple_rnn_cell_7/addAddV2<sequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd:output:0>sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
0sequential_7/simple_rnn_7/simple_rnn_cell_7/TanhTanh3sequential_7/simple_rnn_7/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
7sequential_7/simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
)sequential_7/simple_rnn_7/TensorArrayV2_1TensorListReserve@sequential_7/simple_rnn_7/TensorArrayV2_1/element_shape:output:02sequential_7/simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ`
sequential_7/simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_7/simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿn
,sequential_7/simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¨
sequential_7/simple_rnn_7/whileWhile5sequential_7/simple_rnn_7/while/loop_counter:output:0;sequential_7/simple_rnn_7/while/maximum_iterations:output:0'sequential_7/simple_rnn_7/time:output:02sequential_7/simple_rnn_7/TensorArrayV2_1:handle:0(sequential_7/simple_rnn_7/zeros:output:02sequential_7/simple_rnn_7/strided_slice_1:output:0Qsequential_7/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resourceKsequential_7_simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resourceLsequential_7_simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *6
body.R,
*sequential_7_simple_rnn_7_while_body_93152*6
cond.R,
*sequential_7_simple_rnn_7_while_cond_93151*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
Jsequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
<sequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_7/simple_rnn_7/while:output:3Ssequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0
/sequential_7/simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1sequential_7/simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_7/simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
)sequential_7/simple_rnn_7/strided_slice_3StridedSliceEsequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:08sequential_7/simple_rnn_7/strided_slice_3/stack:output:0:sequential_7/simple_rnn_7/strided_slice_3/stack_1:output:0:sequential_7/simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
*sequential_7/simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          í
%sequential_7/simple_rnn_7/transpose_1	TransposeEsequential_7/simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/simple_rnn_7/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
*sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0¿
sequential_7/dense_7/MatMulMatMul2sequential_7/simple_rnn_7/strided_slice_3:output:02sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_7/dense_7/BiasAddBiasAdd%sequential_7/dense_7/MatMul:product:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_7/dense_7/SigmoidSigmoid%sequential_7/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
IdentityIdentity sequential_7/dense_7/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
NoOpNoOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp*^sequential_7/embedding_7/embedding_lookupC^sequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpB^sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpD^sequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp ^sequential_7/simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2X
*sequential_7/dense_7/MatMul/ReadVariableOp*sequential_7/dense_7/MatMul/ReadVariableOp2V
)sequential_7/embedding_7/embedding_lookup)sequential_7/embedding_7/embedding_lookup2
Bsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpBsequential_7/simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp2
Asequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpAsequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp2
Csequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOpCsequential_7/simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp2B
sequential_7/simple_rnn_7/whilesequential_7/simple_rnn_7/while:c _
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_7_input
æY

G__inference_sequential_7_layer_call_and_return_conditional_losses_94224

inputs5
"embedding_7_embedding_lookup_94107:	ÙE O
=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource:  L
>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource: Q
?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource:  8
&dense_7_matmul_readvariableop_resource: 5
'dense_7_biasadd_readvariableop_resource:
identity¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢embedding_7/embedding_lookup¢5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp¢6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp¢simple_rnn_7/whilej
embedding_7/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿò
embedding_7/embedding_lookupResourceGather"embedding_7_embedding_lookup_94107embedding_7/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_7/embedding_lookup/94107*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
dtype0Î
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_7/embedding_lookup/94107*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¢
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ r
simple_rnn_7/ShapeShape0embedding_7/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:j
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_7/zeros/packedPack#simple_rnn_7/strided_slice:output:0$simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_7/zerosFill"simple_rnn_7/zeros/packed:output:0!simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          º
simple_rnn_7/transpose	Transpose0embedding_7/embedding_lookup/Identity_1:output:0$simple_rnn_7/transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ^
simple_rnn_7/Shape_1Shapesimple_rnn_7/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/Shape_1:output:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÛ
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
simple_rnn_7/strided_slice_2StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask²
4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Æ
%simple_rnn_7/simple_rnn_cell_7/MatMulMatMul%simple_rnn_7/strided_slice_2:output:0<simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ó
&simple_rnn_7/simple_rnn_cell_7/BiasAddBiasAdd/simple_rnn_7/simple_rnn_cell_7/MatMul:product:0=simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¶
6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0À
'simple_rnn_7/simple_rnn_cell_7/MatMul_1MatMulsimple_rnn_7/zeros:output:0>simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Á
"simple_rnn_7/simple_rnn_cell_7/addAddV2/simple_rnn_7/simple_rnn_cell_7/BiasAdd:output:01simple_rnn_7/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#simple_rnn_7/simple_rnn_cell_7/TanhTanh&simple_rnn_7/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ß
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒS
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0simple_rnn_7/zeros:output:0%simple_rnn_7/strided_slice_1:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *)
body!R
simple_rnn_7_while_body_94151*)
cond!R
simple_rnn_7_while_cond_94150*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ò
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0u
"simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿn
$simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
simple_rnn_7/strided_slice_3StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_3/stack:output:0-simple_rnn_7/strided_slice_3/stack_1:output:0-simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskr
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Æ
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_7/MatMulMatMul%simple_rnn_7/strided_slice_3:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_7/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^embedding_7/embedding_lookup6^simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp5^simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp^simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2<
embedding_7/embedding_lookupembedding_7/embedding_lookup2n
5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp2l
4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp2p
6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
²
G__inference_sequential_7_layer_call_and_return_conditional_losses_93942
embedding_7_input$
embedding_7_93926:	ÙE $
simple_rnn_7_93929:   
simple_rnn_7_93931: $
simple_rnn_7_93933:  
dense_7_93936: 
dense_7_93938:
identity¢dense_7/StatefulPartitionedCall¢#embedding_7/StatefulPartitionedCall¢$simple_rnn_7/StatefulPartitionedCallü
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputembedding_7_93926*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_93534¹
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall,embedding_7/StatefulPartitionedCall:output:0simple_rnn_7_93929simple_rnn_7_93931simple_rnn_7_93933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93817
dense_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0dense_7_93936dense_7_93938*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_93664w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp ^dense_7/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:c _
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_7_input
¼,
Ç
while_body_94562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource:  E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢-while/simple_rnn_cell_7/MatMul/ReadVariableOp¢/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¦
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0Ã
while/simple_rnn_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0¾
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ª
 while/simple_rnn_cell_7/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ É
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ}
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

¸
,__inference_simple_rnn_7_layer_call_fn_94271
inputs_0
unknown:  
	unknown_0: 
	unknown_1:  
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93349o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Ü
§
G__inference_sequential_7_layer_call_and_return_conditional_losses_93671

inputs$
embedding_7_93535:	ÙE $
simple_rnn_7_93646:   
simple_rnn_7_93648: $
simple_rnn_7_93650:  
dense_7_93665: 
dense_7_93667:
identity¢dense_7/StatefulPartitionedCall¢#embedding_7/StatefulPartitionedCall¢$simple_rnn_7/StatefulPartitionedCallñ
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_7_93535*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_93534¹
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall,embedding_7/StatefulPartitionedCall:output:0simple_rnn_7_93646simple_rnn_7_93648simple_rnn_7_93650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93645
dense_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0dense_7_93665dense_7_93667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_93664w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp ^dense_7/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ


simple_rnn_7_while_cond_941506
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_28
4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1M
Isimple_rnn_7_while_simple_rnn_7_while_cond_94150___redundant_placeholder0M
Isimple_rnn_7_while_simple_rnn_7_while_cond_94150___redundant_placeholder1M
Isimple_rnn_7_while_simple_rnn_7_while_cond_94150___redundant_placeholder2M
Isimple_rnn_7_while_simple_rnn_7_while_cond_94150___redundant_placeholder3
simple_rnn_7_while_identity

simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


*sequential_7_simple_rnn_7_while_cond_93151P
Lsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_loop_counterV
Rsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_maximum_iterations/
+sequential_7_simple_rnn_7_while_placeholder1
-sequential_7_simple_rnn_7_while_placeholder_11
-sequential_7_simple_rnn_7_while_placeholder_2R
Nsequential_7_simple_rnn_7_while_less_sequential_7_simple_rnn_7_strided_slice_1g
csequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_cond_93151___redundant_placeholder0g
csequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_cond_93151___redundant_placeholder1g
csequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_cond_93151___redundant_placeholder2g
csequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_cond_93151___redundant_placeholder3,
(sequential_7_simple_rnn_7_while_identity
Ê
$sequential_7/simple_rnn_7/while/LessLess+sequential_7_simple_rnn_7_while_placeholderNsequential_7_simple_rnn_7_while_less_sequential_7_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: 
(sequential_7/simple_rnn_7/while/IdentityIdentity(sequential_7/simple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_7_simple_rnn_7_while_identity1sequential_7/simple_rnn_7/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¨	

,__inference_sequential_7_layer_call_fn_93686
embedding_7_input
unknown:	ÙE 
	unknown_0:  
	unknown_1: 
	unknown_2:  
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_93671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_7_input
Ñ8
Í
simple_rnn_7_while_body_941516
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_25
1simple_rnn_7_while_simple_rnn_7_strided_slice_1_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  T
Fsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: Y
Gsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_43
/simple_rnn_7_while_simple_rnn_7_strided_slice_1o
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource:  R
Dsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource: W
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp¢<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ç
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0À
:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0ê
+simple_rnn_7/while/simple_rnn_cell_7/MatMulMatMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¾
;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0å
,simple_rnn_7/while/simple_rnn_cell_7/BiasAddBiasAdd5simple_rnn_7/while/simple_rnn_cell_7/MatMul:product:0Csimple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0Ñ
-simple_rnn_7/while/simple_rnn_cell_7/MatMul_1MatMul simple_rnn_7_while_placeholder_2Dsimple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ó
(simple_rnn_7/while/simple_rnn_cell_7/addAddV25simple_rnn_7/while/simple_rnn_cell_7/BiasAdd:output:07simple_rnn_7/while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)simple_rnn_7/while/simple_rnn_cell_7/TanhTanh,simple_rnn_7/while/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ý
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder-simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒZ
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: À
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: :éèÒ¤
simple_rnn_7/while/Identity_4Identity-simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0^simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
simple_rnn_7/while/NoOpNoOp<^simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp;^simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp=^simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"d
/simple_rnn_7_while_simple_rnn_7_strided_slice_11simple_rnn_7_while_simple_rnn_7_strided_slice_1_0"
Dsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resourceFsimple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"
Esimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"
Csimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resourceEsimple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0"Ü
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2z
;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp;simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2x
:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp2|
<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp<simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
·

Ù
1__inference_simple_rnn_cell_7_layer_call_fn_94770

inputs
states_0
unknown:  
	unknown_0: 
	unknown_1:  
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_93273o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
¨	

,__inference_sequential_7_layer_call_fn_93904
embedding_7_input
unknown:	ÙE 
	unknown_0:  
	unknown_1: 
	unknown_2:  
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_7_layer_call_and_return_conditional_losses_93872o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_7_input
ø

#__inference_signature_wrapper_94243
embedding_7_input
unknown:	ÙE 
	unknown_0:  
	unknown_1: 
	unknown_2:  
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_93225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_7_input
¼,
Ç
while_body_94454
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource:  E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢-while/simple_rnn_cell_7/MatMul/ReadVariableOp¢/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¦
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0Ã
while/simple_rnn_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0¾
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ª
 while/simple_rnn_cell_7/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ É
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ}
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Õ
¥
while_cond_93444
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_93444___redundant_placeholder03
/while_while_cond_93444___redundant_placeholder13
/while_while_cond_93444___redundant_placeholder23
/while_while_cond_93444___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ß	
£
F__inference_embedding_7_layer_call_and_return_conditional_losses_93534

inputs)
embedding_lookup_93528:	ÙE 
identity¢embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
embedding_lookupResourceGatherembedding_lookup_93528Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/93528*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
dtype0ª
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/93528*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æY

G__inference_sequential_7_layer_call_and_return_conditional_losses_94103

inputs5
"embedding_7_embedding_lookup_93986:	ÙE O
=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource:  L
>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource: Q
?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource:  8
&dense_7_matmul_readvariableop_resource: 5
'dense_7_biasadd_readvariableop_resource:
identity¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢embedding_7/embedding_lookup¢5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp¢6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp¢simple_rnn_7/whilej
embedding_7/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿò
embedding_7/embedding_lookupResourceGather"embedding_7_embedding_lookup_93986embedding_7/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_7/embedding_lookup/93986*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
dtype0Î
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_7/embedding_lookup/93986*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¢
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ r
simple_rnn_7/ShapeShape0embedding_7/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:j
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_7/zeros/packedPack#simple_rnn_7/strided_slice:output:0$simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_7/zerosFill"simple_rnn_7/zeros/packed:output:0!simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ p
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          º
simple_rnn_7/transpose	Transpose0embedding_7/embedding_lookup/Identity_1:output:0$simple_rnn_7/transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ^
simple_rnn_7/Shape_1Shapesimple_rnn_7/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/Shape_1:output:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÛ
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒl
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
simple_rnn_7/strided_slice_2StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask²
4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Æ
%simple_rnn_7/simple_rnn_cell_7/MatMulMatMul%simple_rnn_7/strided_slice_2:output:0<simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ °
5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ó
&simple_rnn_7/simple_rnn_cell_7/BiasAddBiasAdd/simple_rnn_7/simple_rnn_cell_7/MatMul:product:0=simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¶
6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0À
'simple_rnn_7/simple_rnn_cell_7/MatMul_1MatMulsimple_rnn_7/zeros:output:0>simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Á
"simple_rnn_7/simple_rnn_cell_7/addAddV2/simple_rnn_7/simple_rnn_cell_7/BiasAdd:output:01simple_rnn_7/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#simple_rnn_7/simple_rnn_cell_7/TanhTanh&simple_rnn_7/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ß
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒS
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0simple_rnn_7/zeros:output:0%simple_rnn_7/strided_slice_1:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_7_simple_rnn_cell_7_matmul_readvariableop_resource>simple_rnn_7_simple_rnn_cell_7_biasadd_readvariableop_resource?simple_rnn_7_simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *)
body!R
simple_rnn_7_while_body_94030*)
cond!R
simple_rnn_7_while_cond_94029*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ò
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0u
"simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿn
$simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
simple_rnn_7/strided_slice_3StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_3/stack:output:0-simple_rnn_7/strided_slice_3/stack_1:output:0-simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskr
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Æ
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_7/MatMulMatMul%simple_rnn_7/strided_slice_3:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_7/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^embedding_7/embedding_lookup6^simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp5^simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp^simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2<
embedding_7/embedding_lookupembedding_7/embedding_lookup2n
5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp5simple_rnn_7/simple_rnn_cell_7/BiasAdd/ReadVariableOp2l
4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp4simple_rnn_7/simple_rnn_cell_7/MatMul/ReadVariableOp2p
6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp6simple_rnn_7/simple_rnn_cell_7/MatMul_1/ReadVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¸
,__inference_simple_rnn_7_layer_call_fn_94282
inputs_0
unknown:  
	unknown_0: 
	unknown_1:  
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0

é
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_94818

inputs
states_00
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
ø 
Ê
while_body_93445
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_simple_rnn_cell_7_93467_0:  -
while_simple_rnn_cell_7_93469_0: 1
while_simple_rnn_cell_7_93471_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_simple_rnn_cell_7_93467:  +
while_simple_rnn_cell_7_93469: /
while_simple_rnn_cell_7_93471:  ¢/while/simple_rnn_cell_7/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
/while/simple_rnn_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_7_93467_0while_simple_rnn_cell_7_93469_0while_simple_rnn_cell_7_93471_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_93393á
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_7/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity8while/simple_rnn_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~

while/NoOpNoOp0^while/simple_rnn_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_7_93467while_simple_rnn_cell_7_93467_0"@
while_simple_rnn_cell_7_93469while_simple_rnn_cell_7_93469_0"@
while_simple_rnn_cell_7_93471while_simple_rnn_cell_7_93471_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2b
/while/simple_rnn_cell_7/StatefulPartitionedCall/while/simple_rnn_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
¼,
Ç
while_body_94346
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource:  E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢-while/simple_rnn_cell_7/MatMul/ReadVariableOp¢/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¦
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0Ã
while/simple_rnn_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0¾
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ª
 while/simple_rnn_cell_7/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ É
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ}
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ù=
¼
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94412
inputs_0B
0simple_rnn_cell_7_matmul_readvariableop_resource:  ?
1simple_rnn_cell_7_biasadd_readvariableop_resource: D
2simple_rnn_cell_7_matmul_1_readvariableop_resource:  
identity¢(simple_rnn_cell_7/BiasAdd/ReadVariableOp¢'simple_rnn_cell_7/MatMul/ReadVariableOp¢)simple_rnn_cell_7/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask
'simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_7_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¬
simple_rnn_cell_7/BiasAddBiasAdd"simple_rnn_cell_7/MatMul:product:00simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_7_matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0
simple_rnn_cell_7/MatMul_1MatMulzeros:output:01simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
simple_rnn_cell_7/addAddV2"simple_rnn_cell_7/BiasAdd:output:0$simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ k
simple_rnn_cell_7/TanhTanhsimple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ö
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_7_matmul_readvariableop_resource1simple_rnn_cell_7_biasadd_readvariableop_resource2simple_rnn_cell_7_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_94346*
condR
while_cond_94345*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ï
NoOpNoOp)^simple_rnn_cell_7/BiasAdd/ReadVariableOp(^simple_rnn_cell_7/MatMul/ReadVariableOp*^simple_rnn_cell_7/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2T
(simple_rnn_cell_7/BiasAdd/ReadVariableOp(simple_rnn_cell_7/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_7/MatMul/ReadVariableOp'simple_rnn_cell_7/MatMul/ReadVariableOp2V
)simple_rnn_cell_7/MatMul_1/ReadVariableOp)simple_rnn_cell_7/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
·

Ù
1__inference_simple_rnn_cell_7_layer_call_fn_94784

inputs
states_0
unknown:  
	unknown_0: 
	unknown_1:  
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_93393o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
Õ
¥
while_cond_93578
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_93578___redundant_placeholder03
/while_while_cond_93578___redundant_placeholder13
/while_while_cond_93578___redundant_placeholder23
/while_while_cond_93578___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ü
§
G__inference_sequential_7_layer_call_and_return_conditional_losses_93872

inputs$
embedding_7_93856:	ÙE $
simple_rnn_7_93859:   
simple_rnn_7_93861: $
simple_rnn_7_93863:  
dense_7_93866: 
dense_7_93868:
identity¢dense_7/StatefulPartitionedCall¢#embedding_7/StatefulPartitionedCall¢$simple_rnn_7/StatefulPartitionedCallñ
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_7_93856*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_93534¹
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall,embedding_7/StatefulPartitionedCall:output:0simple_rnn_7_93859simple_rnn_7_93861simple_rnn_7_93863*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93817
dense_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0dense_7_93866dense_7_93868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_93664w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp ^dense_7/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_dense_7_layer_call_and_return_conditional_losses_93664

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
çD
×
*sequential_7_simple_rnn_7_while_body_93152P
Lsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_loop_counterV
Rsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_maximum_iterations/
+sequential_7_simple_rnn_7_while_placeholder1
-sequential_7_simple_rnn_7_while_placeholder_11
-sequential_7_simple_rnn_7_while_placeholder_2O
Ksequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_strided_slice_1_0
sequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0d
Rsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  a
Ssequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: f
Tsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  ,
(sequential_7_simple_rnn_7_while_identity.
*sequential_7_simple_rnn_7_while_identity_1.
*sequential_7_simple_rnn_7_while_identity_2.
*sequential_7_simple_rnn_7_while_identity_3.
*sequential_7_simple_rnn_7_while_identity_4M
Isequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_strided_slice_1
sequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorb
Psequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource:  _
Qsequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource: d
Rsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢Hsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢Gsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp¢Isequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp¢
Qsequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ©
Csequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0+sequential_7_simple_rnn_7_while_placeholderZsequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0Ú
Gsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOpRsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0
8sequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMulMatMulJsequential_7/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ø
Hsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOpSsequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0
9sequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAddBiasAddBsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul:product:0Psequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Þ
Isequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOpTsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ø
:sequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1MatMul-sequential_7_simple_rnn_7_while_placeholder_2Qsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ú
5sequential_7/simple_rnn_7/while/simple_rnn_cell_7/addAddV2Bsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd:output:0Dsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ «
6sequential_7/simple_rnn_7/while/simple_rnn_cell_7/TanhTanh9sequential_7/simple_rnn_7/while/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ±
Dsequential_7/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_7_simple_rnn_7_while_placeholder_1+sequential_7_simple_rnn_7_while_placeholder:sequential_7/simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒg
%sequential_7/simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ª
#sequential_7/simple_rnn_7/while/addAddV2+sequential_7_simple_rnn_7_while_placeholder.sequential_7/simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_7/simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ï
%sequential_7/simple_rnn_7/while/add_1AddV2Lsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_loop_counter0sequential_7/simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: §
(sequential_7/simple_rnn_7/while/IdentityIdentity)sequential_7/simple_rnn_7/while/add_1:z:0%^sequential_7/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: Ò
*sequential_7/simple_rnn_7/while/Identity_1IdentityRsequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_while_maximum_iterations%^sequential_7/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: §
*sequential_7/simple_rnn_7/while/Identity_2Identity'sequential_7/simple_rnn_7/while/add:z:0%^sequential_7/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ç
*sequential_7/simple_rnn_7/while/Identity_3IdentityTsequential_7/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_7/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: :éèÒË
*sequential_7/simple_rnn_7/while/Identity_4Identity:sequential_7/simple_rnn_7/while/simple_rnn_cell_7/Tanh:y:0%^sequential_7/simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
$sequential_7/simple_rnn_7/while/NoOpNoOpI^sequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpH^sequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpJ^sequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_7_simple_rnn_7_while_identity1sequential_7/simple_rnn_7/while/Identity:output:0"a
*sequential_7_simple_rnn_7_while_identity_13sequential_7/simple_rnn_7/while/Identity_1:output:0"a
*sequential_7_simple_rnn_7_while_identity_23sequential_7/simple_rnn_7/while/Identity_2:output:0"a
*sequential_7_simple_rnn_7_while_identity_33sequential_7/simple_rnn_7/while/Identity_3:output:0"a
*sequential_7_simple_rnn_7_while_identity_43sequential_7/simple_rnn_7/while/Identity_4:output:0"
Isequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_strided_slice_1Ksequential_7_simple_rnn_7_while_sequential_7_simple_rnn_7_strided_slice_1_0"¨
Qsequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resourceSsequential_7_simple_rnn_7_while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"ª
Rsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resourceTsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"¦
Psequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resourceRsequential_7_simple_rnn_7_while_simple_rnn_cell_7_matmul_readvariableop_resource_0"
sequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorsequential_7_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2
Hsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOpHsequential_7/simple_rnn_7/while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2
Gsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOpGsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul/ReadVariableOp2
Isequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpIsequential_7/simple_rnn_7/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
¼,
Ç
while_body_94670
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_7_matmul_readvariableop_resource_0:  G
9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0: L
:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_7_matmul_readvariableop_resource:  E
7while_simple_rnn_cell_7_biasadd_readvariableop_resource: J
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:  ¢.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp¢-while/simple_rnn_cell_7/MatMul/ReadVariableOp¢/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0¦
-while/simple_rnn_cell_7/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_7_matmul_readvariableop_resource_0*
_output_shapes

:  *
dtype0Ã
while/simple_rnn_cell_7/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0¾
while/simple_rnn_cell_7/BiasAddBiasAdd(while/simple_rnn_cell_7/MatMul:product:06while/simple_rnn_cell_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0*
_output_shapes

:  *
dtype0ª
 while/simple_rnn_cell_7/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_7/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
while/simple_rnn_cell_7/addAddV2(while/simple_rnn_cell_7/BiasAdd:output:0*while/simple_rnn_cell_7/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
while/simple_rnn_cell_7/TanhTanhwhile/simple_rnn_cell_7/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ É
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_7/Tanh:y:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ}
while/Identity_4Identity while/simple_rnn_cell_7/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß

while/NoOpNoOp/^while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_7/MatMul/ReadVariableOp0^while/simple_rnn_cell_7/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_7_biasadd_readvariableop_resource9while_simple_rnn_cell_7_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_7_matmul_1_readvariableop_resource:while_simple_rnn_cell_7_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_7_matmul_readvariableop_resource8while_simple_rnn_cell_7_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2`
.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp.while/simple_rnn_cell_7/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_7/MatMul/ReadVariableOp-while/simple_rnn_cell_7/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp/while/simple_rnn_cell_7/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


ó
B__inference_dense_7_layer_call_and_return_conditional_losses_94756

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ø 
Ê
while_body_93286
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_simple_rnn_cell_7_93308_0:  -
while_simple_rnn_cell_7_93310_0: 1
while_simple_rnn_cell_7_93312_0:  
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_simple_rnn_cell_7_93308:  +
while_simple_rnn_cell_7_93310: /
while_simple_rnn_cell_7_93312:  ¢/while/simple_rnn_cell_7/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype0
/while/simple_rnn_cell_7/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_7_93308_0while_simple_rnn_cell_7_93310_0while_simple_rnn_cell_7_93312_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_93273á
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_7/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity8while/simple_rnn_cell_7/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~

while/NoOpNoOp0^while/simple_rnn_cell_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_7_93308while_simple_rnn_cell_7_93308_0"@
while_simple_rnn_cell_7_93310while_simple_rnn_cell_7_93310_0"@
while_simple_rnn_cell_7_93312while_simple_rnn_cell_7_93312_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : 2b
/while/simple_rnn_cell_7/StatefulPartitionedCall/while/simple_rnn_cell_7/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ü
ç
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_93393

inputs

states0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 2
 matmul_1_readvariableop_resource:  
identity

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:  *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ G
TanhTanhadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
Õ
¥
while_cond_93750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_93750___redundant_placeholder03
/while_while_cond_93750___redundant_placeholder13
/while_while_cond_93750___redundant_placeholder23
/while_while_cond_93750___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Õ
¥
while_cond_93285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_93285___redundant_placeholder03
/while_while_cond_93285___redundant_placeholder13
/while_while_cond_93285___redundant_placeholder23
/while_while_cond_93285___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
W
ç
!__inference__traced_restore_94977
file_prefix:
'assignvariableop_embedding_7_embeddings:	ÙE 3
!assignvariableop_1_dense_7_kernel: -
assignvariableop_2_dense_7_bias:)
assignvariableop_3_rmsprop_iter:	 *
 assignvariableop_4_rmsprop_decay: 2
(assignvariableop_5_rmsprop_learning_rate: -
#assignvariableop_6_rmsprop_momentum: (
assignvariableop_7_rmsprop_rho: J
8assignvariableop_8_simple_rnn_7_simple_rnn_cell_7_kernel:  T
Bassignvariableop_9_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel:  E
7assignvariableop_10_simple_rnn_7_simple_rnn_cell_7_bias: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: I
6assignvariableop_15_rmsprop_embedding_7_embeddings_rms:	ÙE @
.assignvariableop_16_rmsprop_dense_7_kernel_rms: :
,assignvariableop_17_rmsprop_dense_7_bias_rms:W
Eassignvariableop_18_rmsprop_simple_rnn_7_simple_rnn_cell_7_kernel_rms:  a
Oassignvariableop_19_rmsprop_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_rms:  Q
Cassignvariableop_20_rmsprop_simple_rnn_7_simple_rnn_cell_7_bias_rms: 
identity_22¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Á

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ç	
valueÝ	BÚ	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_embedding_7_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_7_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_7_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_rmsprop_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_rmsprop_decayIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp(assignvariableop_5_rmsprop_learning_rateIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_rmsprop_momentumIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_rmsprop_rhoIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_8AssignVariableOp8assignvariableop_8_simple_rnn_7_simple_rnn_cell_7_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_9AssignVariableOpBassignvariableop_9_simple_rnn_7_simple_rnn_cell_7_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_10AssignVariableOp7assignvariableop_10_simple_rnn_7_simple_rnn_cell_7_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_15AssignVariableOp6assignvariableop_15_rmsprop_embedding_7_embeddings_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp.assignvariableop_16_rmsprop_dense_7_kernel_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp,assignvariableop_17_rmsprop_dense_7_bias_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_18AssignVariableOpEassignvariableop_18_rmsprop_simple_rnn_7_simple_rnn_cell_7_kernel_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_19AssignVariableOpOassignvariableop_19_rmsprop_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_20AssignVariableOpCassignvariableop_20_rmsprop_simple_rnn_7_simple_rnn_cell_7_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ü3

G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93508

inputs)
simple_rnn_cell_7_93433:  %
simple_rnn_cell_7_93435: )
simple_rnn_cell_7_93437:  
identity¢)simple_rnn_cell_7/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maskã
)simple_rnn_cell_7/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_7_93433simple_rnn_cell_7_93435simple_rnn_cell_7_93437*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_93393n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_7_93433simple_rnn_cell_7_93435simple_rnn_cell_7_93437*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_93445*
condR
while_cond_93444*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
NoOpNoOp*^simple_rnn_cell_7/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : 2V
)simple_rnn_cell_7/StatefulPartitionedCall)simple_rnn_cell_7/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾

'__inference_dense_7_layer_call_fn_94745

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_93664o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ý
²
G__inference_sequential_7_layer_call_and_return_conditional_losses_93923
embedding_7_input$
embedding_7_93907:	ÙE $
simple_rnn_7_93910:   
simple_rnn_7_93912: $
simple_rnn_7_93914:  
dense_7_93917: 
dense_7_93919:
identity¢dense_7/StatefulPartitionedCall¢#embedding_7/StatefulPartitionedCall¢$simple_rnn_7/StatefulPartitionedCallü
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallembedding_7_inputembedding_7_93907*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_93534¹
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall,embedding_7/StatefulPartitionedCall:output:0simple_rnn_7_93910simple_rnn_7_93912simple_rnn_7_93914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_93645
dense_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0dense_7_93917dense_7_93919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_93664w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp ^dense_7/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:c _
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+
_user_specified_nameembedding_7_input"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
X
embedding_7_inputC
#serving_default_embedding_7_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ;
dense_70
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:q
Û
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
µ

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
»

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer

$iter
	%decay
&learning_rate
'momentum
(rho	rmsY	rmsZ	rms[	)rms\	*rms]	+rms^"
	optimizer
J
0
)1
*2
+3
4
5"
trackable_list_wrapper
J
0
)1
*2
+3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_sequential_7_layer_call_fn_93686
,__inference_sequential_7_layer_call_fn_93965
,__inference_sequential_7_layer_call_fn_93982
,__inference_sequential_7_layer_call_fn_93904À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_7_layer_call_and_return_conditional_losses_94103
G__inference_sequential_7_layer_call_and_return_conditional_losses_94224
G__inference_sequential_7_layer_call_and_return_conditional_losses_93923
G__inference_sequential_7_layer_call_and_return_conditional_losses_93942À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÕBÒ
 __inference__wrapped_model_93225embedding_7_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
1serving_default"
signature_map
):'	ÙE 2embedding_7/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_embedding_7_layer_call_fn_94250¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_embedding_7_layer_call_and_return_conditional_losses_94260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
è

)kernel
*recurrent_kernel
+bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;_random_generator
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

>states
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
,__inference_simple_rnn_7_layer_call_fn_94271
,__inference_simple_rnn_7_layer_call_fn_94282
,__inference_simple_rnn_7_layer_call_fn_94293
,__inference_simple_rnn_7_layer_call_fn_94304Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÿ2ü
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94412
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94520
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94628
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94736Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 : 2dense_7/kernel
:2dense_7/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_7_layer_call_fn_94745¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_7_layer_call_and_return_conditional_losses_94756¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
7:5  2%simple_rnn_7/simple_rnn_cell_7/kernel
A:?  2/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel
1:/ 2#simple_rnn_7/simple_rnn_cell_7/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÔBÑ
#__inference_signature_wrapper_94243embedding_7_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
7	variables
8trainable_variables
9regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
ª2§
1__inference_simple_rnn_cell_7_layer_call_fn_94770
1__inference_simple_rnn_cell_7_layer_call_fn_94784¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_94801
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_94818¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
N
	Ptotal
	Qcount
R	variables
S	keras_api"
_tf_keras_metric
^
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
3:1	ÙE 2"RMSprop/embedding_7/embeddings/rms
*:( 2RMSprop/dense_7/kernel/rms
$:"2RMSprop/dense_7/bias/rms
A:?  21RMSprop/simple_rnn_7/simple_rnn_cell_7/kernel/rms
K:I  2;RMSprop/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/rms
;:9 2/RMSprop/simple_rnn_7/simple_rnn_cell_7/bias/rms¥
 __inference__wrapped_model_93225)+*C¢@
9¢6
41
embedding_7_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_7!
dense_7ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_7_layer_call_and_return_conditional_losses_94756\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_7_layer_call_fn_94745O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ»
F__inference_embedding_7_layer_call_and_return_conditional_losses_94260q8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
+__inference_embedding_7_layer_call_fn_94250d8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ç
G__inference_sequential_7_layer_call_and_return_conditional_losses_93923|)+*K¢H
A¢>
41
embedding_7_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
G__inference_sequential_7_layer_call_and_return_conditional_losses_93942|)+*K¢H
A¢>
41
embedding_7_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
G__inference_sequential_7_layer_call_and_return_conditional_losses_94103q)+*@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
G__inference_sequential_7_layer_call_and_return_conditional_losses_94224q)+*@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_7_layer_call_fn_93686o)+*K¢H
A¢>
41
embedding_7_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_7_layer_call_fn_93904o)+*K¢H
A¢>
41
embedding_7_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_7_layer_call_fn_93965d)+*@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_7_layer_call_fn_93982d)+*@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ½
#__inference_signature_wrapper_94243)+*X¢U
¢ 
NªK
I
embedding_7_input41
embedding_7_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_7!
dense_7ÿÿÿÿÿÿÿÿÿÈ
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94412})+*O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 È
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94520})+*O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 Á
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94628v)+*H¢E
>¢;
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 Á
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_94736v)+*H¢E
>¢;
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
  
,__inference_simple_rnn_7_layer_call_fn_94271p)+*O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ  
,__inference_simple_rnn_7_layer_call_fn_94282p)+*O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
,__inference_simple_rnn_7_layer_call_fn_94293i)+*H¢E
>¢;
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
,__inference_simple_rnn_7_layer_call_fn_94304i)+*H¢E
>¢;
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_94801·)+*\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ 
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ 
$!

0/1/0ÿÿÿÿÿÿÿÿÿ 
 
L__inference_simple_rnn_cell_7_layer_call_and_return_conditional_losses_94818·)+*\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ 
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ 
$!

0/1/0ÿÿÿÿÿÿÿÿÿ 
 ß
1__inference_simple_rnn_cell_7_layer_call_fn_94770©)+*\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ 
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ 
"

1/0ÿÿÿÿÿÿÿÿÿ ß
1__inference_simple_rnn_cell_7_layer_call_fn_94784©)+*\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ 
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ 
"

1/0ÿÿÿÿÿÿÿÿÿ 