--[[
nn.Maxout(dimIndex,nGroup)
args:
dimIndex,the index of specify dimension need calculate Maximum value
nGroup,the number of conv groups. 

note: we use this layer followed the layer of nn.DepthConcat, nn.ConcatTable() -> nn.JointTable()

]]
------------

local MaxOut, parent = torch.class('nn.MaxOut', 'nn.Module')
function MaxOut:__init(dimIndex,nGroup)
  parent.__init(self)
  self.nGroup = nGroup or 1
  self.groupSize = nil
  self.input_ = torch.Tensor()
  self.gradInput_ = torch.Tensor()
  self.dimIndex = dimIndex or 1
  self.dimension = 5
end

function MaxOut:_lazyInit()
   self._output = self._output or self.output.new()
   self._indices = self._indices or
      (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())
end

function MaxOut:updateOutput(input)
  if input:dim() ~= 4 then
    error('< the input must be 4 D>')
  end
  if self.groupSize == nil then
 	self.groupSize = input:size(self.dimIndex) / self.nGroup
  end
  if self.input_ == nil or self.input_:nElement()~=input:nElement() or self.input_:type()~=input:type() then
  	self.input_ = torch.Tensor(input:size(1),self.groupSize,input:size(3),input:size(4),self.nGroup):typeAs(input)
  end
  -- copy input -> input_
  for ii=1,self.nGroup do 
    self.input_[{{},{},{},{},ii}]:copy(input[{{},{(ii-1)*self.groupSize+1,ii*self.groupSize},{},{}}])
  end
  -- Max(last_dim) equal to 5, fixme: how to deal with the problem of cuda
  --self.output:typeAs(input)
  self:_lazyInit()
  self._output:typeAs(self.input_)
  torch.max(self._output, self._indices, self.input_, self.dimension) -- max operator on the 5-th dimension
  if input:dim() > 1 then
    self.output = self._output:select(self.dimension, 1)
  else
    self.output = self._output
  end
  return self.output   
end

function MaxOut:updateGradInput(input,gradOutput)
	self:_lazyInit()
    local gradOutputView
    if self.input_ == nil or self.input_:nElement()~=input:nElement() then
  		self.input_ = torch.Tensor(input:size(1),self.groupSize,input:size(3),input:size(4),self.nGroup)--:typeAs(input)
  	end
    if self.input_:dim() > 1 then 
    	gradOutputView = nn.utils.addSingletonDimension(gradOutput, self.dimension)
   	else
     	gradOutputView = gradOutput
   	end
   	self.gradInput_:resizeAs(self.input_):zero():scatter(self.dimension, self._indices, gradOutputView)
   	if self.gradInput==nil or self.gradInput:type() ~= input:type() or self.gradInput:nElement()~= input:nElement() then
		self.gradInput=torch.Tensor():resizeAs(input):zero():typeAs(input)
	end
	--self.gradInput:resizeAs(self.input):zero():typeAs(input)
   	for ii=1,self.nGroup do
    	     self.gradInput[{{},{(ii-1)*self.groupSize+1,ii*self.groupSize},{},{}}]:copy(self.gradInput_[{{},{},{},{},ii}])
  	end
   	return self.gradInput
end

function MaxOut:type(type)
  -- torch.max expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
  if type == 'torch.CudaTensor' then
    parent.type(self, type)
  else
    -- self._indices must be a LongTensor. Setting it to nil temporarily avoids
    -- unnecessary memory allocations.
    local indices
    indices, self._indices = self._indices, nil
    parent.type(self, type)
    self._indices = indices and indices:long() or nil
  end
  return self
end
