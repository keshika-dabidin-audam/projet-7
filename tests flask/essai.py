from flask import Flask, request
 
app = Flask(__name__)
 
@app.route('/')
def index():
    # Get the value of the 'name' parameter from the URL
    name = request.args.get('name')
     
    # Greet Hello to name provided in the URL Parameter
    return "Hello, {}!".format(name)
   
@app.route('/calculate')
def calculate():
   
  # Get the first operand using request arguments
  a = request.args.get('a')
   
  # Get the Second operand using request arguments
  b = request.args.get('b')
   
  # Get the operation to be perform
  operator = request.args.get('operator')
 
  # Make sure that the request arguments are not empty
  if a and b and operator:
    # Convert the request arguments to integers
    a = int(a)
    b = int(b)
 
    # Perform the requested operation
    if operator == 'add':
      result = a + b
    elif operator == 'subtract':
      result = a - b
    elif operator == 'multiply':
      result = a * b
    elif operator == 'divide':
      result = a / b
 
    return f'{a} {operator} {b} = {result}'
  else:
    return 'Error: Insufficient arguments'
 
  app.run()