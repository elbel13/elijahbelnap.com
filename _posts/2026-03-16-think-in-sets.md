---
layout: post
title: "Think in Sets"
author: Elijah Belnap
tags: sql
category: sql
date: 2026-03-16
---

<img src="{{ "/assets/images/think-in-sets.jpeg" | relative_url }}" alt="Think in Sets" style="max-width: 400px; display: block; margin: 0 auto 1rem;" />

Often I see developers using non-ANSI SQL syntax that's borrowed from iterative programming languages. The most common example is the `!=` syntax to show inequality. Given that this does the same thing as its `<>` alternative, why does it matter which one you use?

One reason why I prefer syntax that's ANSI SQL syntax is it reminds me to think in terms of sets and set operations. When programming iteratively it's common to focus on looping through data structures to process values one by one, however relational databases aren't designed to operate this way. A relational database is designed to operate on sets and set based logic rather than iterative operators.

When you join two sets, you aren't iterating through them and processing the rows one by one. What you're logically doing is creating a new set based on the join condition. You shouldn't think in terms of individual rows, although joins may operate on individual rows at times, but rather in terms of the data set as a whole. This set based mentality will help you to write SQL that properly describes the data you're querying for in a way that frees the database engine to retrieve it in the most efficient way possible.

## The For Loop Example

One great example of this is the `FOR` loop. In some databases, you can use a `FOR` loop to iterate through a result set and process each row one by one. However, this is not the most efficient way to retrieve data from a database. Instead of using a `FOR` loop, you should think in terms of sets and use set based operations to retrieve the data you need.

Consider the following example that deletes users that have not logged in for over a year:

```sql
FOR row IN (SELECT id FROM users WHERE last_active < NOW() - INTERVAL '1 year') LOOP
    DELETE FROM users WHERE id = row.id;
END LOOP;
```

This code is inefficient because it retrieves all inactive users and then processes each row one by one to delete them. Instead, you should use a set-based operation to delete all matching rows in a single statement:

```sql
DELETE FROM users
WHERE last_active < NOW() - INTERVAL '1 year';
```

## The For Loop Example in Action

To see the difference concretely, I ran both approaches against a PostgreSQL table of 100,000 users, 20,000 of whom had been inactive for over a year. You can reproduce it with the following setup:

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    last_active TIMESTAMPTZ
);

-- 100k users: 80% recently active, 20% inactive for over a year
INSERT INTO users (username, last_active)
SELECT
    'user_' || i,
    CASE
        WHEN i % 5 = 0 THEN NOW() - INTERVAL '2 years'
        ELSE NOW() - INTERVAL '3 months'
    END
FROM generate_series(1, 100000) AS i;

CREATE INDEX idx_users_last_active ON users(last_active);
ANALYZE users;
```

Running `EXPLAIN (ANALYZE, BUFFERS)` on the set-based delete produces a single, unified plan:

```
Delete on users  (cost=227.55..1215.13 rows=0 width=0) (actual time=11.685..11.686 ms)
  ->  Bitmap Heap Scan on users
        ->  Bitmap Index Scan on idx_users_last_active
              Index Cond: (last_active < now() - '1 year')
              Index Searches: 1
```

One plan. One index search. The planner knows it's operating on a set of 20,000 rows and can optimize accordingly — reading relevant index pages in bulk and deleting them in a single pass.

The iterative loop requires two distinct plans: one for the driving `SELECT`, and then a separate plan executed once per row for the delete:

```
-- The driving SELECT (once):
Bitmap Heap Scan on users
  ->  Bitmap Index Scan on idx_users_last_active
        Index Searches: 1

-- The per-row DELETE (×20,000):
Delete on users
  ->  Index Scan using users_pkey on users
        Index Cond: (id = $1)
        Index Searches: 1
```

That's 20,001 total plans and 20,001 index searches instead of one. The real-world timing reflected this:

| Approach | Time |
|---|---|
| Set-based `DELETE` | **13.7 ms** |
| Iterative PL/pgSQL loop | **110.9 ms** |

**~8× slower** — and the gap only grows with more rows or across a network connection where each individual delete incurs a round-trip.

## Are Control Structures Always Bad?

Control structures like loops and iterators are powerful tools, even in a relational database. However, they should be used judiciously and only when necessary. In most cases, you can achieve the same result using set-based operations, which are more efficient and easier to read.

For example, I often use loops for processing data in batches. However, even when using an iterative control structure I try to stick to set based operations and logic whenever possible. I might be using a loop, but I still operate over a batch of 1000 rows or so at a time. In this way I like to think of it as iterating over subsets within the same set rather than iterating over individual rows.

## Key Takeaways

By thinking in sets rather than iteratively, you allow your database engine to leverage effective data access according to your specific schema and data. Relational database engines are very advanced pieces of software and using set logic is one of the best ways to leverage their strengths.

It may seem small, but I would recommend sticking to ANSI SQL syntax as much as possible to maintain this mindset. Avoid loops and iterators and instead operate on batches of rows. Even small habits such as these can have huge impacts.

Next time you write a query, stop and take a moment to ensure you think in sets.
