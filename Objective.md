You are an expert scientific animator, mathematical educator, and Python developer specializing in Manim.

Your task is to create a **4–5-minute mathematical animation explaining my research paper**, in the visual and pedagogical style of 3Blue1Brown.

The project directory contains:

- `paper.pdf`: My NeurIPS 2026 paper on Multiplexed Factor Analysis (MFA).
- `repository/`: The accompanying Python implementation.

Read the manuscript carefully before developing the animation.

### Objective

Produce a visually compelling, scientifically accurate explanation of the motivation, mathematical framework, and principal results of MFA.

The target audience consists of researchers in machine learning, physics, and neuroscience who may not be familiar with neural burst coding.

Prioritize visual intuition over technical completeness, but preserve the essential mathematical ideas.

### Visual style

Use Manim to produce animations inspired by 3Blue1Brown.

Visual requirements:

- Dark background.
- Clean mathematical typography.
- Consistent colors representing different neural variables.
- Smooth transformations between mathematical expressions and geometric representations.
- Animated neural populations, spike trains, latent spaces, and probability distributions.
- Minimal static text.
- Geometric explanations of otherwise abstract mathematical relationships.
- Smooth camera movements and transitions.
- 16:9 aspect ratio, 1920 × 1080 resolution, and 30 FPS for the final render.

Avoid simply displaying figures or equations from the manuscript. Instead, reconstruct their underlying ideas as animations.

### Narrative structure

Develop the video around the following concepts.

**1. The problem with spike counts**

Introduce how neural activity is conventionally analyzed through firing rates and population activity.

Animate a neural population producing action potentials.

Show how counting spikes can obscure the temporal organization of activity.

Introduce the distinction between isolated spikes and bursts.

**2. The idea of multiplexed neural codes**

Explain how neural activity might contain multiple information channels.

Distinguish between:

- Event rate: How frequently individual events occur.
- Burst probability: The probability that an event is a burst.

Use animated spike trains to illustrate how these quantities can vary independently.

**3. The central mathematical idea of MFA**

Introduce two sets of latent factors governing event rates and burst probabilities.

Show the factors geometrically as low-dimensional population subspaces.

Animate how factors are transformed through loading matrices into the activity of individual neurons.

Explain the generative model intuitively, introducing the relevant Poisson and Binomial distributions.

Use equations from the paper and ensure that their mathematical notation is correct.

**4. Why conventional methods can miss burst structure**

Illustrate how analyzing spike counts alone can obscure factors associated with bursting.

Explain the distinction between the base and hybrid MFA models.

Introduce the null model and explain how it tests whether observed fluctuations in bursting can be predicted from event-rate activity alone.

**5. Principal results**

Explain the main simulation results, emphasizing when the underlying latent structure can be recovered.

Then explain the applications to the VBN and Rodgers datasets.

Recreate simplified, animated versions of the most important results using the actual data or numerical results provided in the paper.

Emphasize that the hybrid model improves the prediction of bursting relative to the null model.

**6. Scientific implications**

Conclude by explaining how MFA combines population analysis with the study of temporal neural codes.

Explain why identifying burst-related latent factors could help investigate neural computations associated with perception, attention, and learning.

Do not suggest that the paper demonstrates a causal role for bursts or proves that bursts encode particular behavioral variables.

### Scientific accuracy

- Use the manuscript as the primary scientific reference.
- Do not invent results, claims, equations, or numerical values.
- Distinguish between demonstrated results and proposed interpretations.
- Preserve the mathematical definitions used in the paper.
- Where simplifications are necessary for visual explanation, state them explicitly.
- Use the accompanying repository to reproduce results whenever feasible.

### Production workflow

**Phase 1: Storyboard**

Before writing animation code, create a detailed storyboard containing:

1. Scene number and title.
2. Duration.
3. Narration script.
4. Mathematical concepts introduced.
5. Description of the animation.
6. Transitions between scenes.
7. Required code, data, or assets.

The total narration should be approximately 550–700 words.

Save the storyboard as `storyboard.md`.

**Stop after this phase and request my approval.**

**Phase 2: Animation**

After storyboard approval, implement the scenes using Manim Community Edition.

Write modular, readable Python code.

Render individual scenes at low resolution first, checking them for errors.

Once the animations are correct, render them at full resolution.

**Phase 3: Narration and synchronization**

Prepare narration scripts for ElevenLabs.

Determine scene durations from the actual generated narration.

Synchronize animations, equations, and transitions to the spoken explanations.

**Phase 4: Final production**

Combine all animations and narration into one video using FFmpeg.

Ensure that:

- The audio and video are synchronized.
- Transitions are smooth.
- Mathematical notation is readable.
- Labels and colors are consistent.
- The final output is a single MP4 file.

Begin by reading the paper and creating the storyboard. Do not begin writing animation code until the storyboard has been reviewed.