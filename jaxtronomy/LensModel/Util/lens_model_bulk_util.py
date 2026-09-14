from functools import partial


def create_unique_lens_model_list(lens_model_list, profile_kwargs_list):
    """From a lens model list and corresponding profile kwargs list, construct new lists
    that only contain unique lens models with unique corresponding profile kwargs. Also
    construct an index list which points each entry of the lens model list to its
    corresponding position on the unique lens model list.

    :param lens_model_list: list of strings with lens model names
    :param profile_kwargs_list: list of dicts, keyword arguments used to initialize
        profile classes in the same order of the lens_model_list
    :return: unique lens model list, unique profile kwargs list, index list
    """
    unique_lens_model_list = []
    unique_profile_kwargs_list = []
    index_list = []

    for i in range(len(lens_model_list)):
        model = lens_model_list[i]
        profile_kwargs = profile_kwargs_list[i]

        model_indices = find_target_indices_in_list(unique_lens_model_list, model)
        profile_kwargs_indices = find_target_indices_in_list(
            unique_profile_kwargs_list, profile_kwargs
        )

        # Check if the model with corresponding profile kwargs already exists in the unique lists
        intersection = list(set(model_indices) & set(profile_kwargs_indices))

        # doesn't exist in the unique lists yet
        if len(intersection) == 0:
            unique_lens_model_list.append(model)
            unique_profile_kwargs_list.append(profile_kwargs)
            index_list.append(len(unique_lens_model_list) - 1)

        # already exists in the unique lists
        else:
            index_list.append(intersection[0])

    return unique_lens_model_list, unique_profile_kwargs_list, index_list


def find_target_indices_in_list(list, target):
    """Finds all indices of a list where target is present.

    :param list: list to search
    :param target: target to find in list
    :return: A list of indices where the target has been found
    """
    indices = []
    for i, element in enumerate(list):
        if element == target:
            indices.append(i)
    return indices


def select_kwargs(profile, params):
    """Returns a callable function that calculates deflection angles after down-
    selecting the relevant kwargs for a given lens model profile.

    :param profile: Instance of a lens model profile e.g. SIE()
    :param params: list of parameter names corresponding to profile e.g. ["theta_E",
        "e1", "e2"]
    :return: A callable function with function signature `derivative(x, y, all_kwargs)`
        where all_kwargs is a dictionary of kwargs that can also include parameters from
        other profiles
    """

    def derivative_wrapper(x, y, all_kwargs, params):
        return profile.derivatives(x, y, *[all_kwargs[param] for param in params])

    return partial(derivative_wrapper, params=params)
