import torch
from train import CPT
from train import get_batch, train_numbers, test_interpo_numbers, test_extrapo_numbers, val_interpo_numbers
from train import detokenize, DIGITS, DEVICE, judge_results

model = CPT()
model.to(DEVICE)

checkpoint = torch.load('./model_and_optimizer_08_04_2023_val_max.ckpt')
model.load_state_dict(checkpoint['model'])

model.eval()

# show 10 examples of each: train, val interpo, val extrapo
def judge_results(numbers, howmany, show=True):
  network_inputs, target = get_batch(numbers, howmany)
  probs, _ = model(network_inputs)
  results = probs.argmax(-1)
  results = [detokenize(result.cpu().numpy()) for result in results]
  results = [result[DIGITS*2+1:-1] for result in results]
  checks = [network_input.endswith(result) for network_input, result in zip(network_inputs, results)]
  print(f"correct: {sum(checks)} out of {howmany}")
  print("first 10 as demonstration: ")
  if not show: return
  for i in range(10):
    network_input, result, correct = network_inputs[i], results[i], checks[i]
    color = "\033[92m" if correct else "\033[91m"
    print(f"{color}input with solution: {network_input} / result: {result} / correct: {correct}\033[0m")


print('results for training numbers: ')
judge_results(train_numbers, 1000, model)

print('results for validation interpolation numbers: ')
judge_results(val_interpo_numbers, 340, model)

print('results for test intrapolation numbers: ')
judge_results(test_interpo_numbers, 1000, model)

print('results for test extrapolation numbers: ')
judge_results(test_extrapo_numbers, 1000, model)
