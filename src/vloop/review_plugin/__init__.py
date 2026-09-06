def register(plugin):
    from vloop.review_operators import CompleteReview, ExcludeReview, StartReview

    for operator in (StartReview, CompleteReview, ExcludeReview):
        plugin.register(operator)
