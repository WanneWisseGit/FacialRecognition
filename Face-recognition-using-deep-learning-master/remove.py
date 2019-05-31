from string import digits

a = ["wanne1","wanne2","wanne3","wanne4","jasper1","jasper2"]
for i in a:
    index = a.index(i)
    remove_digits = str.maketrans('', '', digits)
    res = i.translate(remove_digits)
    a[index] = res
    
print(a)