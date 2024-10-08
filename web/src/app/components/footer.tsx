import { FC } from "react";

export const Footer: FC = () => {
  return (
    <div className="text-center flex flex-col items-center text-xs text-zinc-700 gap-1">
      <div className="text-zinc-400">
        答案由大型语言模型生成，请仔细检查其正确性。
      </div>
    </div>
  );
};
