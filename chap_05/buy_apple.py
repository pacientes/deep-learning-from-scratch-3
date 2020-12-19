import layer_naive

apple = 100
apple_num = 2
tax = 1.1

# layers
mul_apple_layer = layer_naive.MulLayer()
mul_tax_layer = layer_naive.MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(f"forward result {price}")

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(f"backward result {dapple}, {dapple_num}, {dtax}")