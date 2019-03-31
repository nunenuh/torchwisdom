import torchvision.models.squeezenet as squeezenet

model = squeezenet.squeezenet1_1(num_classes=50)
x = torch.randn(2,3,224,224)

x = model.features(x)
print(x.shape)

x = model.classifier(x)
print(x.shape)

x = x.view(x.size(0), 50)
print(x.shape)
