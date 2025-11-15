"""
# File: Project1.py
# Description: 
# Student Names: Katharine Smith and Isabelle Svendsen
# Student EIDs: kbs2529, iks249
# Course Name: CS 313E
# Unique Number: 
# Date Created: 11/5/2025
# Date Last Modified: 
"""
import pandas as pd
import re
from collections import namedtuple, defaultdict
import heapq
from typing import List, Dict, Tuple, Optional


Person = namedtuple("Person", [
    "age", "gender", "education_level", "education_rank",
    "job_title", "years_of_experience", "salary"
])

class DataCleaner:
    DegreeOrder = {
        "none": 0,
        "high school": 1,
        "highschool": 1,
        "bachelor's": 2,
        "bachelors": 2,
        "bachelor's degree": 2,
        "master's": 3,
        "masters": 3,
        "master's degree": 3,
        "phd": 4,
        "doctorate": 4
    }

    def __init__(self, filename: str):
        self.filename = filename

    def clean_degree(self, deg: str) -> Tuple[str, int]:
        if deg is None:
            return ("None", 0)
        d = deg.strip().lower()
        return (deg, self.DegreeOrder.get(d, 0))

    def read_csv(self) -> List[Person]:
        df = pd.read_fwf(self.filename)          # single reliable call
        df.columns = [c.strip() for c in df.columns]

        # safe guard: ensure expected columns present
        expected = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            print("Warning: missing columns:", missing)

        # coerce types for numeric columns if present
        if "Age" in df.columns:
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0).astype(int)
        if "Years of Experience" in df.columns:
            df["Years of Experience"] = pd.to_numeric(df["Years of Experience"], errors="coerce").fillna(0.0)
        if "Salary" in df.columns:
            df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0.0)

        people = []
        for _, row in df.iterrows():
            edu_label = str(row.get("Education Level", "")).strip()
            edu_clean, edu_rank = self.clean_degree(edu_label)
            person = Person(
                age=int(row.get("Age", 0)),
                gender=str(row.get("Gender", "")).strip(),
                education_level=edu_clean,
                education_rank=edu_rank,
                job_title=str(row.get("Job Title", "")).strip(),
                years_of_experience=float(row.get("Years of Experience", 0.0)),
                salary=float(row.get("Salary", 0.0))
            )
            people.append(person)

        return people
    
    
# ---------- 1) Build gender lists (data structure: list) ----------
def split_by_gender(people: List[Person]) -> Tuple[List[Person], List[Person]]:
    males, females = [], []
    for p in people:
        g = p.gender.strip().lower()
        if g.startswith("m"):
            males.append(p)
        elif g.startswith("f"):
            females.append(p)
    return males, females

# ---------- 2) Aggregates by (gender, edu_rank) (data structure: dict) ----------
def avg_salary_by_gender_edu(people: List[Person]) -> Dict[Tuple[str,int], float]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    for p in people:
        key = (p.gender.strip().lower(), p.education_rank)
        sums[key] += p.salary
        counts[key] += 1
    return {k: sums[k]/counts[k] for k in sums}

# ---------- 3) Top-K using a heap (data structure: heap) ----------
def top_k_by_salary(people: List[Person], k: int) -> List[Person]:
    """Return top k persons by salary (largest). Uses a min-heap of size k."""
    if k <= 0:
        return []
    heap = []  # will store (salary, person)
    for p in people:
        if len(heap) < k:
            heapq.heappush(heap, (p.salary, p))
        else:
            # if current salary is larger than smallest in heap, replace
            if p.salary > heap[0][0]:
                heapq.heapreplace(heap, (p.salary, p))
    # convert to sorted descending list
    return [t[1] for t in sorted(heap, key=lambda x: x[0], reverse=True)]

# ---------- 4) Divide & Conquer: merge_sort by salary (algorithm: merge sort) ----------
def merge_sort_by_salary(people: List[Person]) -> List[Person]:
    if len(people) <= 1:
        return people[:]
    mid = len(people) // 2
    left = merge_sort_by_salary(people[:mid])
    right = merge_sort_by_salary(people[mid:])
    # merge two sorted lists (descending by salary)
    i = j = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i].salary >= right[j].salary:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

# ---------- 5) Linked list (recursive insertion) for highest-paid people ----------
class Node:
    def __init__(self, person: Person):
        self.person = person
        self.next: Optional["Node"] = None

class LinkedList:
    def __init__(self):
        self.head: Optional[Node] = None

    def insert_sorted_recursive(self, person: Person):
        """Insert to keep descending salary order using recursion."""
        self.head = self._insert_rec(self.head, person)

    def _insert_rec(self, node: Optional[Node], person: Person) -> Node:
        if node is None:
            return Node(person)
        if person.salary > node.person.salary:
            newnode = Node(person)
            newnode.next = node
            return newnode
        node.next = self._insert_rec(node.next, person)
        return node

    def to_list(self) -> List[Person]:
        out = []
        cur = self.head
        while cur:
            out.append(cur.person)
            cur = cur.next
        return out

# ---------- Example pipeline function tying the above together ----------
def simplified_analysis_pipeline(people: List[Person], top_k: int = 10):
    # 1) split by gender (list)
    males, females = split_by_gender(people)

    # 2) averages (dict)
    averages = avg_salary_by_gender_edu(people)

    # 3) top-k via heap (heap)
    top_male = top_k_by_salary(males, top_k)
    top_female = top_k_by_salary(females, top_k)

    # 4) (optional) show stable sort using merge sort for full ranking
    males_sorted = merge_sort_by_salary(males)
    females_sorted = merge_sort_by_salary(females)

    # 5) build linked lists (recursion) of top-5 for each gender
    male_ll = LinkedList()
    female_ll = LinkedList()
    for p in top_male:
        male_ll.insert_sorted_recursive(p)
    for p in top_female:
        female_ll.insert_sorted_recursive(p)

    return {
        "males_sorted": males_sorted,
        "females_sorted": females_sorted,
        "male_top_k": top_male,
        "female_top_k": top_female,
        "male_ll_top_k": male_ll,
        "female_ll_top_k": female_ll,
        "averages": averages,
    }

def main():
    csv_file = "Salary_Data.csv"  # put your file in same folder or give full path
    cleaner = DataCleaner(csv_file)
    people = cleaner.read_csv()
    print(f"Loaded {len(people)} people from {csv_file}\n")

    results = simplified_analysis_pipeline(people, top_k=5)

    print("=== Top 5 males by salary ===")
    for p in results["male_top_k"]:
        print(f"{p.gender:8} | {p.education_level:16} | ${p.salary:,.2f}")

    print("\n=== Top 5 females by salary ===")
    for p in results["female_top_k"]:
        print(f"{p.gender:8} | {p.education_level:16} | ${p.salary:,.2f}")

    print("\n=== Average salary by (gender, edu_rank) ===")
    for (gender, rank), avg in sorted(results["averages"].items()):
        print(f"{gender:8} edu_rank={rank}: ${avg:,.2f}")

if __name__ == "__main__":
    main()
