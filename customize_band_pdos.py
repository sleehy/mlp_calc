def customize(fig):
    band_axes = [
        ax for ax in fig.axes
        if (ax.get_gid() or "").startswith("band_")
    ]
    pdos_axis = next(
        ax for ax in fig.axes
        if ax.get_gid() == "projected_dos"
    )

    for ax in [*band_axes, pdos_axis]:
        ax.set_ylim()

    band_axes[0].set_ylabel("Phonon frequency (THz)")
    pdos_axis.set_xlabel("Element-projected DOS")
    fig.suptitle("Customized band structure and PDOS")
