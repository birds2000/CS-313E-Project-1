# CS-313E-Project-1
## Katharine Bird Smith and Izzy Svendsen

### (1-2)
We are Data Science majors, and with that, we do a lot of reading files and seeing what we can tell about the characteristics of the file. For this project, we were interested in how gender and level of degree affect the salary a person receives. We wanted to see if a man with a smaller level of education was paid more than a woman with a higher level of education. To do this, we uploaded the Salary_data.csv dataset. This data set has the age of the person, gender, education level, job title, and their current salary. To access this, it is in our Github repository under Salary_data.csv; just download it and make sure it is in the same folder as our Python file. 

### (3)
Our design for the project is made up of models, like our Person tuple and its fields (with age, gender, education_level, etc.) fields and our Node class for the linked list we implemented. To have canonical maps and default filenames, we utilize degree_order to lowercase keys and degree_canonical to make the education level system easier to understand. Our DataCleaner class is helpful in transforming the raw data set rows into canonical Person objects. DataCleaner.readcsv() returns a list/iterator of Person.

### (4)
In terms of libraries, we used:
#### Pandas, to read and coerce tables (CSV). 
#### Heapq, to implement memory-efficient top-k.
#### Pathlib, to reliable cross-platform path handling.
#### Collections (namedtuple, defaultdict), for Person and aggregators.
#### Typing, to type hints (optional but helpful). 

### (5)
For data structures, we used:
#### Person, a namedtuple representing one row.
#### Lists, for storing people when the dataset fits memory or for returning sorted results.
#### Dict / defaultdict, for sums and counts keyed by (gender, education_rank).
#### Min-heap (heapq), keep top-k per group with O(k) memory.
#### Linked list (Node, LinkedList), demonstration of recursive insertion and ordered storage.

### (6)
Since it's a dataset, we don’t have multiple test cases to make sure that it is correct. How do you think we add this in? Is there a different way other than having multiple test cases? The only thing we can think of is separating the dataset into smaller parts so that you are able to see if the answer you got was truly the highest paid salary. However, we feel like this takes away from our main goal. 

### (7)
In terms of the current expectation of the software, we believe that it works well, especially the parts with the material we learned in class, like the linked lists and the classes. The only thing that is holding us behind is getting the code to read a CSV file properly. We had difficulty with this, but we think we got it figured out; we would like your opinion on whether there are easier ways to have it read properly and still have the code run smoothly! 
