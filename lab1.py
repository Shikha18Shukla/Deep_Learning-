s1=float(input("Enter Marks of first subject: "))
s2=float(input("Enter Marks of second subject: "))
s3=float(input("Enter Marks of third subject: "))
s4=float(input("Enter Marks of forth subject: "))
s5=float(input("Enter Marks of fifth subject: "))

total= s1 + s2 + s3 + s4 + s5
average= total/5

max_marks=500
percentage=(total/max_marks)*100

print("\n....Results...")
print("Total Marks:", total)
print("Average Marks:", average)
print("Total Percentage: ",percentage,"%")
