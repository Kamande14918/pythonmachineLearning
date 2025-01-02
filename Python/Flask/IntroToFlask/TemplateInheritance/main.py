# app.py

# import Template from jinja2 for passing the content
from jinja2 import Template

# variables that contain placeholder data
name = 'John'
email = 'you@example.co'


# Create one external form_template html page and read it
File = open('index_template.html', 'r')
content = File.read()
File.close()

# Render the template and pass the variables
template = Template(content)
rendered_form = template.render(pl_name=name, pl_email=email)


# save the txt file in the form.html
output = open('index.html', 'w')
output.write(rendered_form)
output.close()
