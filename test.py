import synapse

arr1 = synapse.NDArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype="f32")
arr2 = synapse.NDArray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [2, 3], dtype="f32")



arr1_sum = arr1.sum()
arr2_sum = arr2.sum()



print(f"arr1_sum: {arr1_sum}")
print(f"arr2_sum: {arr2_sum}")



print(arr1)
print(arr2)

