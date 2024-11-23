from torchvision.models import resnet152
import torch
import coremltools
from PIL import Image
import os
import torchvision.transforms as transforms
import time

resnet152 = resnet152(pretrained=True)
resnet152.eval()

input_shape = (1, 3, 224, 224)
traced_model = torch.jit.trace(resnet152, torch.randn(input_shape))

traced_model.save("resnet152.pt")

traced_model = torch.jit.load("resnet152.pt")
input_shape = (1, 3, 224, 224)
mlmodel = coremltools.convert(
    traced_model,
    source='pytorch',
    inputs=[coremltools.ImageType(name='image', shape=input_shape)],
)

mlmodel.save("resnet152.mlpackage")

mlmodel = coremltools.models.MLModel('resnet152.mlpackage')

image_folder = 'cats_dogs/'

count = 0
test_images = []

for root, dirs, files in os.walk(image_folder):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            image = Image.open(image_path)

            image = image.resize((224, 224))

            test_images.append(image)

            count += 1

            if count % 100 == 0:
                print(f'Processed {count} images')

print(f'Processed {count} images')


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


resnet152_images = []
for image in test_images:
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    resnet152_images.append(input_batch)
    
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')
    
    
start_time = time.time()
for image in test_images:
    mlmodel.predict({'image': image})

end_time = time.time()
total_time = end_time - start_time
print(f'Elapsed time for CoreML [RESNET152]: {total_time:.2f} seconds')


start_time = time.time()
with torch.no_grad():
    for image in resnet152_images:
        pred = resnet152(image)
end_time = time.time()

total_time = end_time - start_time
print(f'Elapsed time for torch [RESNET152]: {total_time:.2f} seconds')