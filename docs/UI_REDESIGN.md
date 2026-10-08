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
- **Screenshot.** `docs/images/ui-redesign/nl1.png`: the real app, offscreen at 1920 x 1080, with a viewer
  image and a dozen nodes of several families. The project label there ("hot_pour / comp_v012") is set
  for the picture only. The offscreen run has no GL, so the GPU pill reads CPU there.

### NL2: node graph look and wires

- **Families.** `nodecatalog.node_family(kind)` gives every registered node type one of the thirteen
  families. The catalog's categories are the families; the plumbing under "Other" (Dot, Group, Backdrop,
  NoOp ...) is filed by `nodecatalog.OTHER_KIND_FAMILY` (neutral Metadata, or Image for ContactSheet and
  PostageStamp). `theme.family_color(kind)` is its colour. `tests/test_graph_look.py` fails on any node
  type without a family.
- **Nodes.** A node is a rounded card (10 px corner; 3D nodes keep their pill and circle shapes): dark
  interior tinted with the family colour, a 2 px outline in the colour, a faint inner glow, a highlight
  along the top edge, a soft halo and the family icon in the colour beside a light name, then a hairline
  and a monospace details line. The details line shows the label and state when there are any and
  otherwise the first knobs, those changed from their defaults first. A selected node has a brighter
  halo and ring. The Viewer wears the interface accent. Postage stamps get rounded corners. The halo is
  a separate child item, so placement and framing see the card alone. At far zoom the glow is skipped.
  Name and details are cut with an ellipsis from the font's own metrics.
- **Wires.** A wire takes its source node's family colour with a faint glow. Mask wires and the Viewer
  tap are dashed (the tap thin and faint). Default shape is the curve of the layout mockup.
- **Port directions.** The main output faces down and the main input up; the A input faces left and a
  mask input faces right (every node that accepts a mask: Grade, Blur and the rest), so a mask wire
  enters it horizontally from the right. Rim sockets of the circle nodes face outward. A wire leaves and
  enters along the direction its port faces in both modes (`graphlook.wire_path`).
- **Curved / Right angle.** The toggle sits in the graph's bottom-right corner next to the zoom control
  and the minimap (`nodebased/graphcorner.py`). Right angle draws horizontal and vertical runs with
  slightly rounded 90 degree turns; on a drop longer than 80 px the horizontal run sits 22 px above the
  input. The choice is saved in the settings (`interface/wire_mode`) and restored at start.
- **Minimap and zoom.** The corner holds a minimap (one rectangle per node in its family colour, the
  view outlined in the accent; click or drag to move the view) and a zoom control (minus, the current
  percentage which resets to 100%, plus).
- **Tests changed because the design changes them.** `test_3d_node_shapes`: a 2D card now has a rounded
  corner, so the square-corner assertion became "rounded card, 3D pill rounder still".
  `test_desktop` title test: the name is 13 px, left aligned after the icon, not 14 pt centred; the
  Viewer-tap test accepts the mockup's custom dash pattern.
