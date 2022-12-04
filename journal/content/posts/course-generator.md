+++
title = "Project-based course generator"
date = 2022-11-21T00:00:00+00:00
lastmod = 2022-12-04T17:28:39+00:00
draft = false
+++

Thoughts about a system which suggests you project ideas and potential course of
action based on your background and preferences.

<!--more-->


## Problems {#problems}


### Not enough space {#not-enough-space}

From personal experience one of the more efficient ways to learn is by building
something from scratch. Most of the books and courses I came across either not
provide practice at all, or provide it by the means of exercises, or provide it
by walking you through the learning project, not giving you enough space for
your own thoughts and experiments. Even if you try to go through the provided
learning project by yourself, your solution quickly diverges from the author's,
and it becomes difficult to follow the course.


### Coming up with a project idea {#coming-up-with-a-project-idea}

That's often the case that you want to learn a new technology, either to apply
for a job or to stay relevant as a professional or just for the fun of it, but
don't have any particular project on your mind. You may want to learn a new
programming language or dive into an entire field like bioinformatics, and you
don't know what would be a good project to start with, but you may have a vague
idea of what you'd like it to be about. Say, you were really fond of taxonomy in
high school, or you were always fascinated by compilers. A relevant project idea
can be a motivator on your learning journey.


## Solution {#solution}


### Tell what and how you'd like to build {#tell-what-and-how-you-d-like-to-build}

You enter a prompt, which can be as vague as `how to make a website` or as precise
as `how to train a diffusion model using PyTorch`, and you get a set of cards
corresponding to specific learning projects. You are free to expand the project
cards and learn more about each of them, before you choose the most appealing to
you. Each project guides you through 2 dimensions:

-   domain knowledge required to build the project;
-   knowledge of developer tools and approaches.

For example, it can suggest you to build a multi-coin crypto wallet in Flutter,
a mobile game of chess in Swift, a reinforcement learning neural net playing
Mario in Python.


### Choose appropriate level of detail {#choose-appropriate-level-of-detail}

Each project is a step-by-step guide where you can vary the level of detail of
each step to provide yourself with a challenging yet approachable task:

-   if you feel like the guide tells you too much and gives you too little space
    to think about the problem on your own, you decrease level of detail and get
    more involved with the task;
-   if you're stuck on a problem for hours or days, you increase level of detail
    and get a hint.

One way to think about it could be a gym: if you're too easy on yourself you
won't get expected results, and if you try too hard you can either get an injury
or inflict on yourself as much pain as to despise the whole enterprise.


### [bonus] Provide bio {#bonus-provide-bio}

Free-form description of your past experiences can be a useful starting point
for the app to both suggest you new projects and to choose the initial level of
detail.


## Users {#users}


### STEM {#stem}

Programming is the main target, but I hope to add mathematics and theoretical
physics too, replacing project steps with exercises. Experimental physics could
be added given the project is build inside of a software simulator.


### Somewhat experienced students and mentors {#somewhat-experienced-students-and-mentors}

I believe that a lot of people who are just starting out in the field need more
than a guide: they need a mentor, and I feel like a mentor boils down to AGI. So
in this case I'd rather concentrate on being a useful tool for humans mentors.
In case of highly motivated beginners and somewhat experienced users though,
this app should be useful by itself.
