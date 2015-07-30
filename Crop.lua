local Crop, Parent = torch.class('nn.Crop', 'nn.Module')
---- forward: two paras, work with nnGraph. need redefine forward

function Crop:__init()
	Parent.__init(self)
	self.mask = torch.Tensor()
end

function Crop:updateOutput(input_data,input)
-- input_data: bottom data
-- input: the current layer's input, format:nBatch x nChanel x H x W  or  nChannel x H x W
	--self.output:resizeAs(input_data)  --:copy(input_data)
	if input_data:dim() == 3 then -- one image
		self.output:resize(input:size(1),input_data:size(2),input_data:size(3))  --:copy(input_data)
		self.mask = torch.Tensor(1,4)  -- pad_l, pad_r,pad_t,pad_b
		self.mask[{1,1}]=math.floor((input:size(2) - input_data:size(2))/2)
		self.mask[{1,2}]=(input:size(2)-input_data:size(2)) - self.mask[{1,1}]
		self.mask[{1,3}]=math.floor((input:size(3) - input_data:size(3))/2)
		self.mask[{1,4}]=(input:size(3) - input_data:size(3)) - self.mask[{1,3}]	
		-- update: crop input
		self.output:copy(input[{{},{self.mask[{1,1}]+1,self.mask[{1,1}]+input_data:size(2)},{self.mask[{1,3}]+1,self.mask[{1,3}]+input_data:size(3)}}])

	elseif input_data:dim() == 4 then  -- batch
		self.output:resize(input:size(1),input:size(2),input_data:size(3),input_data:size(4))  --:copy(input_data)
		--self.mask = torch.Tensor(input_data:size(1),4) 
		self.mask = torch.Tensor(1,4)  -- pad_l, pad_r,pad_t,pad_b
		self.mask[{1,1}]=math.floor((input:size(3) - input_data:size(3))/2)
		self.mask[{1,2}]=(input:size(3)-input_data:size(3)) - self.mask[{1,1}]
		self.mask[{1,3}]=math.floor((input:size(4) - input_data:size(4))/2)
		self.mask[{1,4}]=(input:size(4) - input_data:size(4)) - self.mask[{1,3}]
		-- update: crop input
		self.output:copy(input[{{},{},{self.mask[{1,1}]+1,self.mask[{1,1}]+input_data:size(3)},{self.mask[{1,3}]+1,self.mask[{1,3}]+input_data:size(4)}}])
        else
		error('<Crop updateOutput> illegal input, must be 3 D or 4 D')
	end
	-- updateOutput
	return self.output
end


function Crop:updateGradInput(input,gradOutput)
	--self.gradInput = torch.Tensor()
	if gradOutput:dim() == 3 then
		self.gradInput:resize(gradOutput:size(1),gradOutput:size(2)+self.mask[{1,1}]+self.mask[{1,2}],gradOutput:size(3)+self.mask[{1,3}]+self.mask[{1,4}])
		self.gradInput:fill(0)
		self.gradInput[{{},{self.mask[{1,1}]+1,self.mask[{1,1}]+gradOutput:size(2)},{self.mask[{1,3}]+1,self.mask[{1,3}]+gradOutput:size(3)}}]:copy(gradOutput)
	elseif gradOutput:dim() == 4 then
		self.gradInput:resize(gradOutput:size(1),gradOutput:size(2),gradOutput:size(3)+self.mask[{1,1}]+self.mask[{1,2}],gradOutput:size(4)+self.mask[{1,3}]+self.mask[{1,4}])
		self.gradInput:fill(0)
		self.gradInput[{{},{},{self.mask[{1,1}]+1,self.mask[{1,1}]+gradOutput:size(3)},{self.mask[{1,3}]+1,self.mask[{1,3}]+gradOutput:size(4)}}]:copy(gradOutput)
	else
		error('<crop updateGradInput> illegal gradOutput, must be 3 D or 4 D')
	end 
	return self.gradInput
end

function Crop:forward(input_data, input)
-- rewrite forward, need be fixed given the input data format
   return self:updateOutput(input_data,input)
end
