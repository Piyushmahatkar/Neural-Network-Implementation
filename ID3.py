

import math
from random import *
import sys

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.leaf = 0
        self.parent = None
        self.begin = -1
        self.end = -1
        self.attr_name = ""
        self.node_num = -1
        self.rclass = -1
        self.rootcount = -1
        self.neg = -1
        self.height = 0
        self.train_data=[[]]
        self.count=0

        def set_attr_name(s):
            self.attr_name = s

        def set_left(l):
            self.__left = l

        def set_right(r):
            self.__right = r

        def set_parent(p):
            self.__parent = p

        def set_neg(n):
            self.__neg = n

        def get_neg(self):
            return self.__neg



class dtree:
    def __init__(self):
        self.troot = None
        self.count = 0
        self.pruned_nodes = 0
        self.pruned_leaf_nodes = 0
        self.flag = 0
        self.examples = [[]]
        self.features = []
        self.no_atr = 20
        self.leaf_count = 0
        self.train_data=[[]]
        self.found = 0


    def decisionTree(self, begin, end, no_atr, root):
        root.begin = begin
        root.end = end
        if root.parent is not None:
            root.height = root.parent.height + 1

        pure_class = self.pure(begin, end)
        if pure_class == 1:
            root.rclass = 1
            root.leaf = 1
            return

        if pure_class == 0:
            root.rclass = 0
            root.leaf = 1
            return
        if root.leaf != 1:
            e = []
            for i in range(no_atr):
                en = self.calc_ent(begin, end, i)
                e.append(en)
            best_feature = self.min_ig(e)
            negatives = self.calculate_negative_sample(begin, end, best_feature)
            root.neg = negatives
            self.filter(begin, end, best_feature, negatives)
            root.attr_name = self.features[best_feature]
            self.count += 1
            root.count = self.count
            root.left = Node()
            root.right = Node()
            root.left.parent = root
            root.right.parent = root
            if begin == 0:
                 self.decisionTree(begin, root.neg, self.no_atr, root.left)
            else:
                 self.decisionTree(begin, begin + root.neg, self.no_atr, root.left)
            self.decisionTree(begin + root.neg, end, self.no_atr, root.right)



# function to interchange two tuples of the list

    def interchange(self, a, b):
        self.examples[a], self.examples[b] = self.examples[b], self.examples[a]

# function to filter the rows based on positive and negative examples of the selected attribute

    def filter(self, begin, end, col, zeroes):
        zeroes += begin
        first = begin
        last = end-1
        while first < zeroes and last > zeroes-1:
            while int(self.examples[first][col]) == 0:
                first += 1
            while int(self.examples[last][col]) == 1:
                last -= 1
            self.interchange(first, last)
        if int(self.examples[zeroes-1][col]) > int(self.examples[zeroes][col]):
            self.interchange(zeroes, zeroes-1)
# function to calculate the negative samples in the samples of a specific attribute

    def calculate_negative_sample(self, begin_row, end_row, attribute_num):
        count0 = 0
        for i in range(begin_row, end_row):
            if int(self.examples[i][attribute_num]) == 0:
                count0 += 1
        return count0
# checking if the partition is pure

    def pure(self, begin, end):
        sum = 0
        #count1 = 1
        for i in range(begin, end):
           sum+=int(self.examples[i][self.no_atr])
        if sum==0:
            return 0
        elif sum==end-begin:
                return 1
        else:
            return -1

# Function for finding information gain

    def information_gain(self, begin, end, attr):
        parent_entropy = self.base_entropy(begin, end, attr)
        IG = []
        for i in range(attr):
                ig = self.calc_ent(begin, end, i)
                IG.append(ig)
        ind = self.min_ig(IG)
        return ind

# This function takes a list of information gains and  returns index of the element with max IG

    def min_ig(self, l):
        min_info = l[0]
        for x in range(0, len(l)):
            if min_info > l[x]:
                 min_info = l[x]
        index = l.index(min_info)
        return index

# Function gives the entropy of the parent

    def base_entropy(self, begin, end, att_index):
        positive_count = 0
        negative_count = 0
        for i in range(begin, end):
            if int(self.examples[i][att_index]) == 0:
                negative_count += 1
            else:
                positive_count += 1
        entropy = self.getEntropy(positive_count, negative_count)
        return entropy


# Calculates entropy
    def calc_ent(self, begin, end, attr):
        pos1 = 0
        pos0 = 0
        neg1 = 0
        neg0 = 0
        ent0 = 0
        ent1 = 0
        for i in range(begin, end):
            if int(self.examples[i][attr]) == 0:
                if int(self.examples[i][self.no_atr]) == 0:
                    neg0 += 1
                else:
                    pos0 += 1
            else:
                if int(self.examples[i][self.no_atr]) == 0:
                    neg1 += 1
                else:
                    pos1 += 1
        ent0 = self.getEntropy(pos0, neg0)
        ent1 = self.getEntropy(pos1, neg1)
        total = ((pos0 + neg0) / len(self.examples)) * ent0 + ((pos1 + neg1) / len(self.examples) * ent1)
        return total

    def getEntropy(self, p, n):

        sum = p + n
        if sum == 0:
            entropy = 0
            return entropy
        pos = p / sum
        neg = n / sum

        if pos != 0:
            plog = math.log2(pos)
        else:
            plog = 0
        if neg != 0:
            nlog = math.log2(neg)
        else:
            nlog = 0
        entropy = -(pos * plog) - (neg * nlog)
        return entropy



    def print_tree(self, root):
        height=0
        root.rootcount = 0
        if root.rclass == -1:
            if root.left.attr_name != "" and root.right.attr_name != "":
                i = 0
                while i < root.height:
                    print("|", end="")
                    i += 1
                print(root.attr_name + "=" + str(root.rootcount))
                root.rootcount += 1
            if root.left.attr_name == "" and root.right.attr_name == "":
                i = 0
                while i < root.height:
                    print("|", end="")
                    i += 1
                print(root.attr_name + "=" + str(root.rootcount) + ":" + str(root.left.rclass))
                self.leaf_count += 1  # change this
            elif root.left.attr_name == "":
                i = 0
                while i < root.height:
                    print("|", end="")
                    i += 1
                print(root.attr_name + "=" +str(root.rootcount) + ":" + str(root.left.rclass))
                root.rootcount += 1
                i = 0
                while i < root.height:
                    print("|", end="")
                    i += 1
                print(root.attr_name + "=" + str(root.rootcount))
                root.rootcount += 1
            elif root.right.attr_name == "":
                i = 0
                while i < root.height:
                    print("|", end="")
                    i += 1
                print(str(root.attr_name) + "=" + str(root.rootcount) + ":" + str(root.right.rclass))
                root.rootcount += 1
                i = 0
                while i < root.height:
                    print("|", end="")
                    i += 1
                print(root.attr_name + "=" + str(root.rootcount))
                root.rootcount += 1
            self.print_tree(root.left)
            if root.left.attr_name == "" and root.right.attr_name == "":
                self.print_tree(root.right)
            else:
                if root.rootcount < 2:
                    i = 0
                    while i < root.height:
                        print("|", end="")
                        i += 1
                    print(root.attr_name + "=" + str(root.rootcount))
                self.print_tree(root.right)



    def traversal(self, root, node):

        if self.found !=1:
            if root.count == node:
                root.right = None
                root.left = None
                if root.neg > (root.end - root.begin - root.neg):
                    root.attr_name = ""
                    root.rclass = 0
                else:
                    root.attr_name = ""
                    root.rclass = 1
                self.found = 1
            else:
                if root.left is not None:
                    self.traversal(root.left, node)
                if root.right is not None:
                    self.traversal(root.right, node)



            # calculations before pruning

    def pruning(self, pruned_nodes, root):
        #print(pruned_nodes)
        node = []
        for i in range(0, pruned_nodes):
            x = randint(1, self.count)  # radint is a inbuilt function in python which generates random nos
            node.append(x)

        for i in range(0, len(node)):
            num = node[i]
            self.traversal(root, num)


    def ImportData(self, x):
        self.train_data.clear()
        if x==0:
            f_test = open(sys.argv[1], 'r')
            for line in f_test:
                line = line.strip("\r\n")
                self.train_data.append(line.split(','))
            f_test.close()


            #self.train_data.remove([])  # removing first element of the list
            attr = self.train_data[0]  # taking the first row into the attributes list
            self.train_data.remove(attr)

        if x==1:
            f_test = open(sys.argv[2], 'r')
            for line in f_test:
                line = line.strip("\r\n")
                self.train_data.append(line.split(','))
            f_test.close()
            #self.train_data.remove([])  # removing first element of the list
            attr = self.train_data[0]  # taking the first row into the attributes list
            self.train_data.remove(attr)
        if x==2:
            self.train_data.clear()
            f_test = open(sys.argv[3], 'r')
            for line in f_test:
                line = line.strip("\r\n")
                self.train_data.append(line.split(','))
            f_test.close()
            #self.train_data.remove([])  # removing first element of the list
            attr = self.train_data[0]  # taking the first row into the attributes list
            self.train_data.remove(attr)


    def pruning_aftermath1(self, root):
        if root.attr_name == "":
            self.pruned_nodes += 1
        if root.left is not None:
            self.pruning_aftermath1(root.left)
        if root.right is not None:
            self.pruning_aftermath1(root.right)

    def pruning_aftermath2(self, root):
        if root.left is None and root.right is None:
            self.pruned_leaf_nodes += 1
        if root.left is not None:
            self.pruning_aftermath2(root.left)
        if root.right is not None:
            self.pruning_aftermath2(root.right)

    def accuracy(self, root):
        var = 0

        for i in range(len(self.train_data)):
            t = root
            while t.attr_name != "":
                feat = t.attr_name
                index = self.features.index(feat)
                if int(self.train_data[i][index]) == 0:
                    t = t.left
                    # print("Left")
                else:
                    # print("right")
                    t = t.right
            if int(self.train_data[i][self.no_atr]) == t.rclass:
                var += 1
        return var / len(self.train_data) * 100


    def printing(self, root):
        self.ImportData(0)
        print("Number of training instances =" + str(len(self.train_data)))  # modify later on
        print("Number of training attributes =" + str(len(self.features)))
        print("Total number of nodes in the tree =" +str(root.count))
        print("Number of leaf nodes in the tree =" +str(self.leaf_count))
        print("Accuracy of the model on the training dataset =" +str(self.accuracy(root)) +"%")
        print()
        print()
        self.ImportData(1)
        print("Number of test instances = ", len(self.train_data))  # modify later on
        print("Number of test attributes", len(self.features))
        print("Accuracy of the model on the validation dataset before pruning =", self.accuracy(root))
        print()
        print()
        self.ImportData(2)
        print("Number of test instances = ", len(self.train_data))  # modify later on
        print("Number of test attributes", len(self.features))
        print("Accuracy of the model on the Test dataset before pruning =", self.accuracy(root))

    def after_pruning(self, root):
        self.pruning_aftermath1(root)
        self.pruning_aftermath2(root)
        self.ImportData(0)
        print("Number of training instances =" + str(len(self.train_data)))
        print("Number of training attributes =" + str(len(self.features)))
        print("Total number of nodes in the tree =" + str(self.pruned_nodes))
        print("Number of leaf nodes in the tree =" + str(self.leaf_count-20))
        print("Accuracy of the model on the training dataset =" + str(self.accuracy(root))+"%")
        print()
        self.ImportData(1)
        print("Number of training instances =" + str(len(self.train_data)))
        print("Number of training attributes =" + str(len(self.features)))
        print("Accuracy of the model on the test dataset =" + str(self.accuracy(root)) + "%")
        print()

        self.ImportData(2)
        print("Number of training instances =" + str(len(self.train_data)))
        print("Number of training attributes =" + str(len(self.features)))
        print("Accuracy of the model on the test dataset =" + str(self.accuracy(root)) + "%")



def main():
    f_train = open(sys.argv[1], 'r')
    train_data = [[]]
    for line in f_train:
        line = line.strip("\r\n")
        train_data.append(line.split(','))
    f_train.close()
    train_data.remove([])  # removing first element of the list
    attr = train_data[0]  # taking the first row into the attributes list
    train_data.remove(attr)

    features = attr[:-1]  # removing the last column 'class '
    obj = dtree()
    main_root = Node()
    obj.examples = train_data
    obj.features = features
    obj.decisionTree(0, len(obj.examples), 20, main_root)
    obj.print_tree(main_root)
    #train_accuracy=obj.accuracy(main_root)
    #print(len(obj.examples))
    print("Pre Prune")
    print("---------------------------------------------------")
    obj.printing(main_root)
    print("Enter pruning factor")
    pfactor=float(input())
    x=obj.count*pfactor

    obj.pruning(int(x), main_root)
    print("Post Prune")
    print("---------------------------------------------------")
    obj.after_pruning(main_root)
    #obj.print_tree(main_root)




if __name__ == '__main__':
        main()
