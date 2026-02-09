FAQ
===

How do I choose a colormap quickly?
-----------------------------------

Use this rule of thumb:

- Sequential: ordered values (low to high).
- Diverging: values around a meaningful center.
- Qualitative: categories.
- Circular: phase and angle.

Why should I avoid ``jet``?
---------------------------

``jet`` is not perceptually uniform. It can create visual boundaries that are
not present in data.

How do I make plots safer for colorblind readers?
-------------------------------------------------

Assess the colormap first with ``assess_cmap`` and review the colorblind
simulation panels before finalizing figures.

Do I always need to uniformize a colormap?
------------------------------------------

No. Start with assessment, then apply uniformization only if you detect
lightness non-linearity, asymmetry, or obvious artifacts.
