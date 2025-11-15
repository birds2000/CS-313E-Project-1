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
from pathlib import Path
from collections import namedtuple, defaultdict
import heapq
from typing import List, Dict, Tuple, Optional

import pandas as pd

Person = namedtuple(
    "Person",
    [
        "age",
        "gender",
        "education_level",
        "education_rank",
        "job_title",
        "years_of_experience",
        "salary",
    ],
)

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
        "doctorate": 4,
    }

    # canonical labels to return as education_level (optional)
    DegreeCanonical = {
        0: "None",
        1: "High School",
        2: "Bachelor's",
        3: "Master's",
        4: "PhD",
    }

    def __init__(self, filename: str):
        self.filename = filename

    def clean_degree(self, deg: Optional[str]) -> Tuple[str, int]:
        """
        Normalize degree string and return (label, rank).
        If deg is None or unknown, returns ("None", 0).
        """
        if deg is None:
            return ("None", 0)
        d = str(deg).strip().lower()
        rank = self.DegreeOrder.get(d, 0)
        label = self.DegreeCanonical.get(rank, "None")
        return (label, rank)

    def read_csv(self) -> List[Person]:
        """
        Read the file robustly. Attempts pd.read_csv; falls back to pd.read_fwf.
        Returns a list of Person namedtuples.
        """
        p = Path(self.filename)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {self.filename}")

        # try common CSV read first
        read_err = None
        try:
            # allow pandas to infer the delimiter; engine='python' is more forgiving
            df = pd.read_csv(p, engine="python")
        except Exception as e_csv:
            read_err = e_csv
            try:
                # fallback to fixed-width if CSV fails
                df = pd.read_fwf(p)
            except Exception as e_fwf:
                # raise informative error including both exception messages
                raise RuntimeError(
                    f"Failed to read file as CSV (err: {e_csv}) and as fixed-width (err: {e_fwf})."
                )

        # Normalize column names
        df.columns = [c.strip() for c in df.columns]

        expected = [
            "Age",
            "Gender",
            "Education Level",
            "Job Title",
            "Years of Experience",
            "Salary",
        ]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            print("Warning: missing columns:", missing)

        # Coerce numeric columns if they exist
        if "Age" in df.columns:
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0).astype(int)
        if "Years of Experience" in df.columns:
            df["Years of Experience"] = (
                pd.to_numeric(df["Years of Experience"], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
        if "Salary" in df.columns:
            df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0.0)

        people: List[Person] = []
        for _, row in df.iterrows():
            edu_label_raw = row.get("Education Level", "")
            edu_label, edu_rank = self.clean_degree(edu_label_raw)
            try:
                age = int(row.get("Age", 0))
            except Exception:
                age = 0
            gender = str(row.get("Gender", "")).strip()
            job_title = str(row.get("Job Title", "")).strip()
            years_exp = float(row.get("Years of Experience", 0.0))
            salary = float(row.get("Salary", 0.0))

            person = Person(
                age=age,
                gender=gender,
                education_level=edu_label,
                education_rank=edu_rank,
                job_title=job_title,
                years_of_experience=years_exp,
                salary=salary,
            )
            people.append(person)

        return people


# Build gender lists (data structure: list)
def split_by_gender(people: List[Person]) -> Tuple[List[Person], List[Person]]:
    males, females = [], []
    for p in people:
        g = (p.gender or "").strip().lower()
        if g.startswith("m"):
            males.append(p)
        elif g.startswith("f"):
            females.append(p)
    return males, females


# Aggregates by (gender, edu_rank) (data structure: dict)
def avg_salary_by_gender_edu(people: List[Person]) -> Dict[Tuple[str, int], float]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    for p in people:
        key = (p.gender.strip().lower(), p.education_rank)
        sums[key] += p.salary
        counts[key] += 1
    # defensive: avoid ZeroDivisionError (shouldn't happen because sums only created when counts increment)
    averages = {}
    for k in sums:
        if counts[k] == 0:
            averages[k] = 0.0
        else:
            averages[k] = sums[k] / counts[k]
    return averages


# Top-K using a heap (robust tie-breaker)
def top_k_by_salary(people: List[Person], k: int) -> List[Person]:
    """Return top k persons by salary (largest). Uses a min-heap of size k."""
    if k <= 0:
        return []
    heap = []  # will store (salary, idx, person)
    for idx, p in enumerate(people):
        entry = (p.salary, idx, p)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        else:
            if p.salary > heap[0][0]:
                heapq.heapreplace(heap, entry)
    # convert to sorted descending list by salary
    sorted_entries = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
    return [t[2] for t in sorted_entries]


# Divide & Conquer: merge_sort by salary (descending)
def merge_sort_by_salary(people: List[Person]) -> List[Person]:
    if len(people) <= 1:
        return people[:]
    mid = len(people) // 2
    left = merge_sort_by_salary(people[:mid])
    right = merge_sort_by_salary(people[mid:])
    i = j = 0
    merged: List[Person] = []
    while i < len(left) and j < len(right):
        if left[i].salary >= right[j].salary:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


# Linked list (recursive insertion) for highest-paid people
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
        out: List[Person] = []
        cur = self.head
        while cur:
            out.append(cur.person)
            cur = cur.next
        return out


# Example pipeline function tying the above together
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

    # 5) build linked lists (recursion) of top-k for each gender
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
    try:
        people = cleaner.read_csv()
    except Exception as e:
        print("Error reading data:", e)
        return

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
