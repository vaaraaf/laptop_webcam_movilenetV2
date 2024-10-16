import torch
import torchvision
def evaluate_image(image_path,
                   model: torch.nn.Module,
                   transform: torchvision.transforms,
                   dataset_name = 'imagenet', # this is the name of the dataset you are planning to wotk with
                   ):
  from PIL import Image
  # Downloading the name of the classes of the dataset
  if dataset_name == 'imagenet':
    dataset_classes = torchvision.models.EfficientNet_B0_Weights.DEFAULT.meta['categories']
  elif dataset_name == 'cifar10':
    dataset_classes = torchvision.datasets.CIFAR10.classes
  elif dataset_name == 'cifar100':
    dataset_classes = torchvision.datasets.CIFAR100.classes

  # print(len(dataset_classes))
  testing_image = Image.open(image_path)
  testing_image_transformed = transform(testing_image)
  testing_image_transformed_unsqueeze = testing_image_transformed.unsqueeze(dim = 0)
  model.eval()
  with torch.inference_mode():
    prediction_tensor = model(testing_image_transformed_unsqueeze)
    prediction_tensor_softmax = prediction_tensor.squeeze().softmax(dim = 0)
    # print(f'the maximum is {prediction_tensor_softmax.argmax(dim=0)} and the class name is {dataset_classes[prediction_tensor_softmax.argmax(dim=0)]} and value is {prediction_tensor_softmax[prediction_tensor_softmax.argmax(dim=0)]}')
    prediction_tensor_softmax_sorted , indices = torch.sort(prediction_tensor_softmax, descending=True)
    print(f'############   NEW OBJECT IS:   ################\n'
          f'{dataset_classes[indices[0]]}: {round(prediction_tensor_softmax_sorted[0].item()*100, 4)}\n'
          f'{dataset_classes[indices[1]]}: {round(prediction_tensor_softmax_sorted[1].item()*100, 4)}\n'
          f'{dataset_classes[indices[2]]}: {round(prediction_tensor_softmax_sorted[2].item()*100, 4)}\n'
          f'{dataset_classes[indices[3]]}: {round(prediction_tensor_softmax_sorted[3].item()*100, 4)}\n'
          f'{dataset_classes[indices[4]]}: {round(prediction_tensor_softmax_sorted[4].item()*100, 4)}')
    # print(prediction_tensor_softmax_sorted
    # print(torch.sum(prediction_tensor_softmax))
    # print(prediction_tensor)
  # print(f'{testing_image_transformed.shape}')
