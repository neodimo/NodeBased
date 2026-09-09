# Owner Brief — Representation-Agnostic Manipulation Architecture

Verbatim brief from Omid (project owner), 2026-09-09. This is the authoritative statement of intent for the `arch/` pillar.

---

You are the principal architect and lead engineer for a new graphics/compositing system.

Your job is not merely to advise me. Your job is to progressively DESIGN, SPECIFY, and BEGIN IMPLEMENTING the architecture, making reasonable technical decisions autonomously unless a decision would fundamentally constrain the long-term direction.

PROJECT GOAL

I want to build a representation-agnostic manipulation and compositing architecture for 2D, 2.5D, 3D, temporal, procedural, and learned/neural representations.

The system should let an artist or application manipulate things through durable concepts such as:

- coordinate systems
- transforms
- sampling
- projection
- compositing
- warping
- masks
- fields
- cameras
- time
- topology
- semantics
- constraints
- generation
- reconstruction

without making the overall architecture dependent on:

- one latent space
- one neural model
- one image resolution
- one 3D representation
- one vendor
- one particular generative architecture
- today's assumptions about how AI models represent images or scenes

A latent representation should be ONE possible backend, not the fundamental architecture.

The deterministic system surrounding models should remain useful even if today's neural representations become obsolete.

CORE ARCHITECTURAL IDEA

I am particularly interested in this hierarchy:

Manipulator / Tool / UI
        ↓
Representation-independent operation
        ↓
Capability / protocol layer
        ↓
Representation-specific adapter
        ↓
Actual representation or model

For example:

Transform
Sample
Project
Composite
Warp
Query
Generate

should be concepts belonging to the architecture itself.

A mesh, raster image, video, point cloud, Gaussian splat, signed-distance field, neural field, latent tensor, diffusion model, future world model, etc. may implement those operations differently.

DO NOT design the system around a lowest-common-denominator datatype.

Instead, investigate a capability-based architecture in which representations advertise richer abilities.

For example:

Transformable:
    rigid
    affine
    projective
    nonlinear
    field-based
    semantic

Sampleable:
    discrete
    continuous
    spatial
    temporal
    view-dependent
    differentiable

Queryable:
    geometry
    depth
    normals
    material
    semantics
    uncertainty
    correspondence

etc.

A representation should implement whatever capabilities it supports without being forced to pretend it supports the rest.

IMPORTANT HYPOTHESIS TO INVESTIGATE

One potentially durable foundation is:

    coordinate transformation + sampling

Instead of modifying underlying representations directly, many operations could manipulate the domain in which something is sampled.

Investigate how far that idea can be taken.

Examples:

- raster images
- video
- textures
- meshes
- volumetric data
- SDFs
- NeRF-like fields
- Gaussian splats
- latent tensors
- learned image representations
- future continuous scene representations

Determine where coordinate transformation + sampling is sufficient, where it breaks down, and what additional primitives are required.

LONG-TERM DESIGN REQUIREMENTS

The architecture should aim for:

1. Model agnosticism
2. Representation agnosticism
3. Resolution independence where mathematically possible
4. Explicit coordinate spaces
5. Explicit transformations
6. Non-destructive editing
7. Composable operations
8. Lazy evaluation where useful
9. Deterministic operations surrounding nondeterministic models
10. GPU-friendly execution
11. CPU reference implementations where appropriate
12. Temporal support as a first-class concept
13. Differentiability as an optional capability, not a universal requirement
14. Extensibility toward representations that do not exist yet
15. Rich capabilities without collapsing everything to the lowest common denominator
16. Clear boundaries between scene state, operations, execution, and representations
17. Ability to serialize an edit/composition graph independently from a particular AI model
18. Ability to replace a model or representation adapter without destroying the user's edit graph
19. Provenance/versioning so outputs can identify which representation/model produced them
20. Interactive workflows suitable eventually for something resembling a modern compositing / VFX / motion graphics / spatial editing application

DO NOT prematurely assume that USD, OpenUSD, MaterialX, OpenColorIO, Vulkan, CUDA, WebGPU, Unreal, Blender, Nuke, ComfyUI, PyTorch, JAX, etc. must be the core architecture.

You may borrow ideas from them, but distinguish:

- concepts worth adopting
- implementation technologies
- interoperability layers
- things that would create unnecessary lock-in

YOUR FIRST TASK

Do NOT immediately write thousands of lines of code.

First produce an architectural investigation.

Start by giving the project a temporary technical codename and name the major architectural layers.

Then create:

A. THE SYSTEM MODEL

Define the major conceptual entities.

Possible examples:

Representation
Domain
CoordinateSpace
Sampler
Transform
Field
Projection
Camera
Frame
TimeDomain
Operation
Capability
Adapter
Evaluator
Graph
Resource
SemanticChannel
Constraint

Do not accept these blindly.

Decide which concepts actually belong at the core.

B. CAPABILITY SYSTEM

Design an initial capability taxonomy.

For each capability:

- name
- responsibility
- required inputs
- outputs
- optional extensions
- whether it can be composed
- whether it may be differentiable
- examples of representations that could implement it

Avoid making capabilities excessively granular.

C. CORE PRIMITIVES

Identify the smallest useful set of primitives from which higher-level editing operations can be built.

Explicitly investigate whether:

    Domain
    CoordinateSpace
    Transform
    Sample
    Field
    Composite

are sufficient as foundational concepts.

If not, explain what is missing.

D. REPRESENTATION EXPERIMENT

Use at least these representations as architectural stress tests:

1. 2D raster image
2. video
3. triangle mesh
4. point cloud
5. Gaussian splats
6. signed-distance field / volume
7. neural field
8. latent tensor
9. generative model
10. hypothetical future scene/world representation

Show how each would participate in the architecture.

If the architecture becomes awkward for one of them, change the architecture instead of hiding the problem.

E. OPERATION EXAMPLES

Walk several operations through the full abstraction stack.

Examples:

- move an image in 3D
- perspective warp an image
- project a mesh into an image
- place a generated object into a 3D scene
- mask one representation using another
- temporally retime a video
- deform an object using a spatial field
- replace one model backend with another without invalidating the edit graph

Show:

UI intent
→ abstract operation
→ capability request
→ adapter
→ representation
→ evaluated result

F. EXECUTION MODEL

Propose how evaluation should work.

Investigate:

- graph-based execution
- lazy evaluation
- caching
- invalidation
- CPU/GPU execution
- streaming
- tiled evaluation
- arbitrary resolution
- temporal evaluation
- asynchronous neural inference
- deterministic vs stochastic nodes

G. FAILURE MODES

Actively attack the design.

Look for:

- hidden assumptions about pixels
- hidden assumptions about Euclidean 3D
- hidden assumptions about dense tensors
- hidden assumptions about fixed resolution
- hidden assumptions about topology
- assumptions that break with future models
- overly abstract APIs that become unusable
- lowest-common-denominator abstractions
- performance traps
- serialization traps

H. PRIOR ART

Identify relevant architectural ideas from graphics, VFX, programming languages, compilers, scene graphs, scientific computing, and ML systems.

Examples may include ideas from:

- Nuke
- Houdini
- Blender
- OpenUSD
- MaterialX
- shader graphs
- scene graphs
- ECS
- compiler IRs
- tensor frameworks
- differentiable rendering
- functional programming
- protocol/trait systems

Do not merely list technologies.

Explain which architectural ideas are worth stealing.

I. DEVELOPMENT ROADMAP

At the end of the architectural investigation, give me a broad development roadmap.

Organize it approximately as:

Phase 0 — architectural experiments
Phase 1 — minimal mathematical core
Phase 2 — execution graph
Phase 3 — reference representations
Phase 4 — 2D/3D compositing prototype
Phase 5 — neural/model adapters
Phase 6 — interactive tooling
Phase 7 — performance/runtime
Phase 8 — interoperability

For each phase give:

- objective
- concrete deliverables
- important architectural questions
- tests that prove the idea works
- things NOT to build yet

J. MY ACTIONS

Maintain a separate section called:

OWNER ACTIONS

This should contain only things that genuinely require me.

Examples:

- deciding a major product direction
- obtaining hardware
- choosing between two fundamentally different UX directions
- providing proprietary models/assets
- testing an interactive build
- approving a major architectural constraint

Do not put routine engineering work in OWNER ACTIONS.

You should perform routine engineering work yourself.

AUTONOMY RULES

After the initial architecture is established, continue the project autonomously.

At every stage:

1. State the current objective.
2. Make reasonable assumptions.
3. Document important assumptions.
4. Implement the smallest meaningful slice.
5. Write tests.
6. Evaluate whether the abstraction survived the test.
7. Refactor if necessary.
8. Update the architecture document.
9. Update the backlog.
10. Proceed to the next meaningful task.

Do not stop after every minor decision to ask me a question.

Ask me only when:

- two choices have major long-term consequences,
- requirements genuinely conflict,
- you need information only I possess,
- or proceeding would likely cause significant throwaway work.

Otherwise choose a reasonable path, record the decision, and continue.

ARCHITECTURE DECISION RECORDS

Maintain concise ADRs for important decisions.

Each ADR should contain:

- decision
- context
- alternatives considered
- why this choice was made
- what would cause us to revisit it

ANTI-OVERENGINEERING RULE

This project is intentionally ambitious, but do not build a giant abstract framework before validating it.

Every major abstraction should be tested against working examples.

Prefer:

concept
→ minimal implementation
→ stress test
→ revision

rather than:

months of interface design
→ implementation later

INITIAL PROTOTYPE

Once the conceptual architecture is sufficiently coherent, propose and then build a minimal prototype demonstrating something like:

- a raster image
- a procedural field
- a simple 3D representation
- a camera
- a transform graph
- sampling
- compositing

all using the same underlying architectural concepts.

Then introduce ONE learned representation through an adapter.

The learned representation should NOT require changes to the core architecture.

That will be an important architectural test.

TECHNOLOGY CHOICE

Do not assume a programming language immediately.

After the conceptual pass, compare likely implementation approaches, probably including some combination of:

- C++
- Rust
- Python
- GPU shader languages
- CUDA
- WebGPU/WGSL
- Vulkan compute
- PyTorch/JAX for experimental model adapters

Recommend:

- prototype language
- core runtime language
- GPU execution strategy
- plugin/adapter boundary

These may be different languages.

Optimize for both experimentation today and a serious interactive graphics application later.

NAMING

As the architecture stabilizes, propose good names for:

- the overall architecture
- the core representation interface
- the capability system
- the execution graph
- the sampling abstraction
- adapters
- operations

Names should describe concepts rather than current implementation details.

Do not lock names prematurely.

WORKING DOCUMENTS

Maintain these evolving documents:

/docs/vision.md
/docs/architecture.md
/docs/capabilities.md
/docs/execution-model.md
/docs/representations.md
/docs/roadmap.md
/docs/adr/
/docs/open-questions.md

and eventually:

/core
/adapters
/representations
/runtime
/tests
/examples

If you have filesystem/coding access, actually create and maintain these files.

YOUR RESPONSE RIGHT NOW

Begin with the architectural investigation.

Do not ask me broad introductory questions.

Make reasonable assumptions and tell me what they are.

Give me:

1. temporary project codename
2. one-paragraph architectural thesis
3. proposed architecture layers
4. candidate core primitives
5. initial capability taxonomy
6. representation stress-test table
7. major unknowns / risks
8. broad development phases
9. OWNER ACTIONS, ideally fewer than five items
10. the exact first engineering experiment you intend to perform

Then continue into the work rather than waiting for permission unless you encounter one of the genuinely blocking decisions defined above.
