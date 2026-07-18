WEB_DIRECTORY = "./web"


async def comfy_entrypoint():
    from .k2_region_comfy.nodes import K2RegionExtension

    return K2RegionExtension()


__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
