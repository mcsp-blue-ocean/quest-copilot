import Image from "next/image";

export default function Header() {
  return (
    <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
      <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-800 bg-gradient-to-b from-zinc-800 pb-6 pt-8 backdrop-blur-2xl text-slate-100 dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
        💻 Copilot for Software Engineers &nbsp;
        <code className="font-mono font-bold text-xl">🔍 U E S T</code>
      </p>
      <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-black via-gray dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
        <a
          href="https://github.com/mcsp-blue-ocean/quest/"
          className="flex items-center justify-center font-nunito text-lg font-bold gap-2"
        >
          <span className="text-slate-100">Built by Blue Ocean 🌊</span>
          <Image
            className="rounded-xl"
            src="/logo.png"
            alt="QUEST Logo"
            width={80}
            height={40}
            priority
          />
        </a>
      </div>
    </div>
  );
}
