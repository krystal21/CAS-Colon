import torchvision.models as models
import torch

def get_model(model_name, num_classes=10):
    if model_name == "resnet50":
        # 使用 torchvision 加载预训练的 ResNet50 模型
        model = models.resnet50(pretrained=True)
        
        # 修改输出层的类别数
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        return model
    if model_name == "resnet18":
        # 使用 torchvision 加载预训练的 ResNet50 模型
        model = models.resnet18(pretrained=True)
        
        # 修改输出层的类别数
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        return model
    if model_name == "densenet121":
        # 使用 torchvision 加载预训练的 DenseNet121 模型
        model = models.densenet121(pretrained=True)
        
        # 修改输出层的类别数
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        
        return model
    if model_name == "inception":
        # 使用 torchvision 加载预训练的 DenseNet121 模型
        model = models.inception_v3(pretrained=True)
        
        # 修改输出层的类别数
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        return model