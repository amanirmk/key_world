with open('requirements.txt', 'r') as file:
    lines = file.readlines()

new_lines = []
for line in lines:
    if '=' in line:
        package = line.split('=')[0]
        version = line.split('=')[1]

        new_lines.append(package + '=' + version + '\n')
    else:
        new_lines.append(line)

with open('requirements_ubuntu.txt', 'w') as file:
    file.writelines(new_lines)