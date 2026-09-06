import getpass

import fiftyone.operators as foo
import fiftyone.operators.types as types

from .review import change_reviews, config_for_dataset


def selected_samples(ctx):
    return [ctx.current_sample] if ctx.current_sample else list(ctx.selected)


class ReviewAction(foo.Operator):
    action = None
    title = None

    @property
    def config(self):
        return foo.OperatorConfig(
            name=f"review_{self.action}",
            label=self.title,
            description="열린 이미지 또는 선택한 이미지의 검수 상태를 변경합니다.",
            dynamic=True,
            view_target=False,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        ids = selected_samples(ctx)
        message = inputs.str(
            "selection", default=f"검수 대상: {len(ids)}장", view=types.MarkdownView()
        )
        if len(ids) > 100:
            message.invalid = True
            message.error_message = (
                "수동 검수는 한 번에 100장까지 선택하세요. "
                "대량 채택은 vloop review-batch를 사용하세요."
            )
        if not ids or not ctx.dataset or not ctx.dataset.info.get("vloop_review_config"):
            message.invalid = True
            message.error_message = "vloop review로 프로젝트를 열고 이미지를 선택하세요."
        inputs.str("reviewer", label="검수자", required=True, default=getpass.getuser())
        inputs.str("note", label="메모", default="")
        if self.action == "complete":
            inputs.bool(
                "confirm",
                label="선택한 이미지의 저장된 정답을 확인했습니다.",
                required=True,
                default=False,
            )
            inputs.bool(
                "confirm_empty", label="빈 정답이 있다면 객체가 없음을 확인했습니다.", default=False
            )
        return types.Property(inputs, view=types.View(label=self.title))

    def execute(self, ctx):
        if len(selected_samples(ctx)) > 100:
            raise ValueError(
                "수동 검수는 한 번에 100장까지 가능합니다. 대량 채택은 review-batch를 사용하세요."
            )
        if self.action == "complete" and ctx.params.get("confirm") is not True:
            raise ValueError("검수한 정답을 확인한 후 완료해주세요.")
        cfg = config_for_dataset(ctx.dataset)
        result = change_reviews(
            cfg,
            selected_samples(ctx),
            self.action,
            ctx.params.get("reviewer", ""),
            note=ctx.params.get("note", ""),
            confirm_empty=ctx.params.get("confirm_empty") is True,
        )
        ctx.ops.reload_dataset()
        ctx.ops.reload_samples()
        ctx.ops.notify(
            f"검수 상태 변경 {result['changed']}장, 실패 {result['failed']}장",
            variant="warning" if result["failed"] else "success",
        )
        return result

    def resolve_output(self, ctx):
        output = types.Object()
        output.int("changed", label="상태 변경")
        output.int("failed", label="실패")
        output.list("errors", types.Object(), label="실패 원인", view=types.JSONView())
        return types.Property(output)


class StartReview(ReviewAction):
    action = "start"
    title = "VLoop: 검수 시작 / 재검수"


class CompleteReview(ReviewAction):
    action = "complete"
    title = "VLoop: 검수 완료"


class ExcludeReview(ReviewAction):
    action = "exclude"
    title = "VLoop: 검수 제외"
