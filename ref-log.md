Reflection Document – Multi-Agent Travel Planning App

Working on the multi-agent travel planner gave me a better understanding of how two different agents can collaborate to do something more meaningful than either could do alone. I realized that defining each agent’s role clearly is what makes the whole system work. The Planner needed to focus on creativity and user preferences, while the Reviewer had to think more critically and fix mistakes. It felt a bit like having a writer and an editor who speak the same language but have very different goals. Once I structured their outputs carefully — especially the Planner’s day-by-day markdown format and the Reviewer’s “Feasibility / Delta / Revised Itinerary” sections — the back-and-forth became much smoother.

One challenge I ran into was that the Planner sometimes went overboard, suggesting too many activities or impossible travel distances. To fix that, I added more direct instructions about pacing, budgets, and grouping days by city or region. Another tricky part was getting the Reviewer to use the internet tool in a focused way instead of just searching for everything. Adding clear examples in the prompt helped it check the right facts, like museum hours or train times, without wasting calls.

I also emphasized structured Markdown outputs, so Streamlit could render them as clean sections. The workflow highlighted how careful prompt design and role specification can substitute for complex orchestration code.Overall, this project showed me how much of multi-agent design depends on communication — not just code, but also how you phrase instructions and structure the output.

External tools and GenAI assistance:
I used OpenAI’s GPT-4 models and Streamlit to build and test the app. I also used generative AI to help polish some wording and debug minor issues with prompt formatting.
