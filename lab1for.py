def calculate_results(marks):
    total=sum(marks)
    average=total/len(marks)
    return total , average
def main():
    marks=[]
    for i in range(1,6):
        score=float(input("Enter marks of your 5 subjects: "))
        marks.append(score)
        total,average=calculate_results(marks)
    print("\n....Results...")
    print("Total Marks:", total)
    print("Average Marks:", average)
main()

