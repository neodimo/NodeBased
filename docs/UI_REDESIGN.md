# New look (UI redesign)

The approved design is a set of HTML mockups (layout, node graph look and wires). This file records what
each stage of the restyle changed in the real Qt app. Stages: 1 colours, spacing, type and the top bar;
2 node graph look and wires; 3 drag a free node onto a wire to insert it; 4 icons and the left column;
5 viewer strip and timeline; 6 Properties; 7 motion.

## Lane 2 step notes

### NL1: colours, spacing, type and the top bar

- **Tokens.** `nodebased/theme.py` holds the design tokens: `TOKENS` (surfaces `bg0`–`bg3`, hairlines
  `line` / `line2`, four text levels `tx0`–`tx3`, accent `acc` #5ee0b5), `FAMILY_COLORS` (the thirteen
  family colours, no yellow), `RADIUS` (9–12 px on controls and panels), `SPACING` and `TYPE` (Noto Sans,
  13 px base, Noto Sans Mono for values). The default theme is `NodeBased`, built from them; Charcoal,
  Graphite, Slate and Ash remain selectable and pick up the same stylesheet.
- **One stylesheet.** `theme.build_style()` styles the whole application, including the top bar (by
  objectName). Chrome widgets carry no colour literals; `tests/test_topbar.py` scans `topbar.py` for hex
  colours.
- **Top bar.** `nodebased/topbar.py`: logo mark, project / shot with a saved dot (project is the folder,
  shot is the file name; the dot dims while there are unsaved changes), workspace tabs as a segmented
  control, the search box, the GPU pill (adapter name read from the GL renderer, last frame time from the
  viewer status) and the Check for updates button as the primary button. There is no Export button.
- **Menus.** The menu bar is hidden. Its menus sit under the ☰ button in the top bar, together with the
  quick actions the old toolbar carried (Open image, Add node, Save project, Export image, 3D viewport,
  Slice viewer, Cache inspector). Every menu action is also added to the window so its shortcut keeps
  working while the bar is hidden.
- **Search.** The box opens the existing node search under it; Tab in the node graph is unchanged. The
  search lists nodes only for now; commands and settings are not searched yet.
- **Workspace tabs.** Composite and Render show the 2D viewer, 3D and Simulate the 3D viewport. Simulate
  also opens the Slice viewer and Cache inspector, Render opens the Conditioning queue. The tabs never
  close a panel, and the highlighted tab follows the viewer's own 2D / 3D switch.
- **Narrow windows.** Sizes come from font metrics. When the bar is narrower than its contents it drops
  the shot name first, then the GPU adapter name (the light and the frame time stay).
