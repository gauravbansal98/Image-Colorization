
import wget

wget.download("http://colorization.eecs.berkeley.edu/siggraph/models/model.caffemodel", "models/reference_model/model.caffemodel")
wget.download("http://colorization.eecs.berkeley.edu/siggraph/models/global_model.caffemodel", "models/global_model/global_model.caffemodel")
wget.download("http://colorization.eecs.berkeley.edu/siggraph/models/dummy.caffemodel", "models/global_model/dummy.caffemodel")
wget.download("http://colorization.eecs.berkeley.edu/siggraph/models/pytorch.pth", "models/pytorch/pytorch_trained.pth")
wget.download("http://colorization.eecs.berkeley.edu/siggraph/models/caffemodel.pth", "models/pytorch/caffemodel.pth")
