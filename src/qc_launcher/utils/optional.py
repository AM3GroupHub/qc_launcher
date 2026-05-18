from __future__ import annotations


def is_missing_package(exc: ModuleNotFoundError, package_name: str) -> bool:
    missing_name = getattr(exc, "name", "")
    return missing_name == package_name or missing_name.startswith(f"{package_name}.")


def missing_optional_dependency(
    feature: str,
    extra_name: str,
    package_name: str | None = None,
    install_hint: str | None = None,
) -> ModuleNotFoundError:
    package_label = extra_name if package_name is None else package_name
    install_command = f"pip install -e .[{extra_name}]" if install_hint is None else install_hint
    return ModuleNotFoundError(
        f"{feature} requires optional dependency '{package_label}'. "
        f"Install it with `{install_command}`."
    )
