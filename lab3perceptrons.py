
def calculate_weight_sum(inputs,weights,bias):
    if len(inputs)!=len(weights):
        raise ValueError("Nummber of inputs and weights must be equal.")
    weighted_sum=0
    for i in range(len(inputs)):
        weighted_sum+=inputs[i] * weights[i]
    weighted_sum+=bias
    return weighted_sum
input_values=[0.5,0.8,0.2]
weight_values=[0.7,-0.3,0.9]
bias_term=0.1
summation_result=calculate_weight_sum(input_values,weight_values,bias_term)
print(f"The weighted sum is :{summation_result}")
